package aws

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SaveMode, SparkSession, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.collection.mutable

object TopicModeling {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: InputDir OutputDir")
    }

    // Path of input file
    val inputFilePath = args(0)
    // Path of output file
    val outputFilePath = args(1)

    // Initializing spark context object
    val sparkConf = new SparkConf().setAppName("Topic Modeling")
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SparkSession.Builder()
      .config(sparkConf)
      .getOrCreate()
    import sqlContext.implicits._

    // Reading data from text file
    val corpus = sparkContext.textFile(inputFilePath)

    // Defining stop words set
    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet

    // Pre-processing data --> converting to lower case, tokenizing, removing words of length <= 3, removing stop words,
    // and filtering non-letter characters
    val tokenized: RDD[Seq[String]] = corpus.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3)
      .filter(token => !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))

    // calculating word counts
    val termCounts: Array[(String, Long)] = tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    // Removing 20 most frequent words
    val numOfStopWords = 20
    val vocabArray: Array[String] = termCounts.takeRight(termCounts.length - numOfStopWords).map(_._1)

    // Mapping term with term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    // Converting documents to term count vectors
    val documents: RDD[(Long, Vector)] = tokenized.zipWithIndex().map {case (tokens, id) =>
      val counts = new mutable.HashMap[Int, Double]()
      tokens.foreach { term =>
        if(vocab.contains(term)) {
          val idx = vocab(term)
          counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
        }
      }
      (id, Vectors.sparse(vocab.size, counts.toSeq))
    }

    // Setting LDA parameters
    val numOfTopics = 5
    val lda = new LDA().setK(numOfTopics).setMaxIterations(10)

    val ldaModel = lda.run(documents)
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

    // Initialize DF
    val schema = StructType(
      StructField("Topic", StringType, true) ::
        StructField("Word", StringType, true) ::
        StructField("value", StringType, true) :: Nil)
    var initialDF = sqlContext.createDataFrame(sparkContext.emptyRDD[Row], schema)
    var i = 0
    topicIndices.foreach { case (terms, termWeights) =>
      terms.zip(termWeights).foreach { case (term, weight) =>
        initialDF = initialDF.union(Seq(("Topic"+i, s"${vocabArray(term.toInt)}", weight)).toDF)
      }
      i += 1
    }
    // Writing topics to output file
    initialDF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("csv")
      .option("header", "true")
      .save(outputFilePath)
  }
}

// spark-submit --deploy-mode cluster --class aws.TopicModeling s3://aws-logs-884479802072-us-east-1/tweetanalysis_2.11-0.1.jar s3://aws-logs-884479802072-us-east-1/book_text_data.txt s3://aws-logs-884479802072-us-east-1/topic-modeling-output/topics.txt
