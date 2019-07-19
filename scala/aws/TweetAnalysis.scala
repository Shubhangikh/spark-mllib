package aws

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

case class Metric(metricName: String, value: Double)

object TweetAnalysis {
  def main(args: Array[String]) {
    if (args.length != 2) {
      println("Usage: InputDir OutputDir")
    }
    // input directory
    val inputDir = args(0)

    // output directory
    val outputDir = args(1)
    print("input : "  + inputDir + " output : " + outputDir)
    // create Spark context with Spark configuration
    val sparkConf = new SparkConf().setAppName("Tweet Analysis")
    val spark = new SparkSession.Builder()
      .config(sparkConf)
      .getOrCreate()
    import spark.implicits._

    // Reading the tweets dataset from csv and converting it into a DataFrame
    var tweetsOriginal = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv(inputDir)

    // cleaning the text column and removing the irrelevant columns
    val cleanTweetsOriginal = tweetsOriginal.filter(col("text").isNotNull).drop("tweet_coord", "airline_sentiment_gold", "negativereason_gold")

    // splitting the dataset into train and test
    val splits = cleanTweetsOriginal.randomSplit(Array(0.8, 0.2), 24)
    val train = splits(0)
    val test = splits(1)

    // pipeline : stage-1 -> Tokenizer
    val tokenizer = new Tokenizer().setInputCol("text")
      .setOutputCol("words")

    // pipeline : stage-2 -> StopWordsRemover
    val stopWordsRemover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    // pipeline : stage-3 -> HashingTermFrequency
    val hashingTF = new HashingTF().setNumFeatures(1000)
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")

    // pipeline : stage-4 -> String Indexer
    val indexer = new StringIndexer().setInputCol("airline_sentiment")
      .setOutputCol("label")

    // pipeline : stage-5 -> Linear Regression model
    val lr = new LogisticRegression().setMaxIter(10)
      .setRegParam(0.1)

    // pipeline : for a 5-stage operation
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, indexer, lr))

    // ParamGrid to make a list of parameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    // Defining a CrossValidator for finding the best model based on ParamGrid
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(9) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(train)

    // Make predictions on test documents.
    var predictedResults = cvModel.bestModel.transform(test)

    // Compute raw scores on the test set
    val predictionAndLabels = predictedResults.select("prediction", "label").map {
      case Row(label: Double, prediction: Double) => (prediction, label)
    }

    // MulticlassMetrics for getting the evaluation metrics
    val metrics = new MulticlassMetrics(predictionAndLabels.rdd)

    val Metric1 = Metric("Weighted precision", metrics.weightedPrecision)
    val Metric2 = Metric("Weighted recall", metrics.weightedRecall)
    val Metric3 = Metric("Weighted F1 score", metrics.weightedFMeasure)
    val Metric4 = Metric("Weighted false positive rate", metrics.weightedFalsePositiveRate)
    val Metric5 = Metric("Weighted True positive rate", metrics.weightedTruePositiveRate)

    val metricSeq = Seq(Metric1, Metric2, Metric3, Metric4, Metric5)
    val metricsDF = metricSeq.toDF()

    // Write the data from metrics to a csv file
    metricsDF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("csv")
      .option("header", "true")
      .save(outputDir)
  }

}

// spark-submit --deploy-mode cluster --class MyApp s3://com.anurag-wids/project_test_2.11-0.1.jar s3://com.anurag-wids/Tweets.csv s3://com.anurag-wids/tweetsOutput
