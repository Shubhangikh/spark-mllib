package aws

import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object PageRank {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: PageRank -> InputDir iterations OutputDir")
    }

    // input directory
    val inputDir = args(0)

    // number of iterations
    val iterations = args(1).toInt

    // output directory
    val outputDir = args(2)

    // create Spark context with Spark configuration
    val sparkConf = new SparkConf().setAppName("Page Rank")
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SparkSession.Builder()
      .config(sparkConf)
      .getOrCreate()

    import sqlContext.implicits._

    // create a DataFrame from csv data with origin, dest pair data
    val data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", ",").load(inputDir)

    // DataFrame with ORIGIN and DEST columns
    val airportData = data.select("ORIGIN", "DEST")

    // Group inlinks by airport code
    val airportLinks = airportData.map(row => (row.getString(0), row.getString(1))).collect()
    val airportLinksPll = sparkContext.parallelize(airportLinks)
    val airportGroupedLinks = airportLinksPll.groupByKey()

    // Initializing page rank with 10.0 for every node
    var pageRanks = airportGroupedLinks.mapValues(v => 10.0)

    // Running for n iterations
    for (i <- 1 to iterations) {
      val maps = airportGroupedLinks.join(pageRanks).values.flatMap { case (airports, pageRank) =>
        val size = airports.size
        val ratio = pageRank / size
        airports.map(airport => (airport, ratio))
      }
      val nodeCount = airportGroupedLinks.count()
      pageRanks = maps.reduceByKey(_ + _).mapValues((0.15 / nodeCount) + 0.85 * _)
    }

    // Sort page ranks in descending order
    val output = pageRanks.sortBy(_._2, false)

    // Results saved to the output directory
    output.saveAsTextFile(outputDir)
  }
}

// spark-submit --deploy-mode cluster --class aws.PageRank s3://aws-logs-884479802072-us-east-1/tweetanalysis_2.11-0.1.jar s3://aws-logs-884479802072-us-east-1/origin_dest_data.csv 10 s3://aws-logs-884479802072-us-east-1/page-ranks-output
