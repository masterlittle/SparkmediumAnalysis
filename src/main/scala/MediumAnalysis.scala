
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}
/**
  * Created by grofers on 16/11/17.
  */
object MediumAnalysis {

  protected def getSparkSession(appName: String): SparkSession = {
    val spark = SparkSession
      .builder()
      .appName(appName)
      .getOrCreate()

    spark.sparkContext.hadoopConfiguration.set("fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
    spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "XXXX")
    spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "XXXX")
    spark.sparkContext.hadoopConfiguration.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

    spark
  }

  def main(args: Array[String]): Unit = {

    val sparkSession = getSparkSession("Medium")

    val data = sparkSession.sqlContext
      .read
      .json("s3n://test/medium-posts-popular.json")
      .repartition(96, col("postId"))
      .cache()

    val dataInitial: Dataset[Row] = Utils.preProcessTextData(data)

    val (modelInit: PipelineModel, dataset: Dataset[Row]) = prepareForModelling(dataInitial)

    processLDA(modelInit, dataset, sparkSession)

  }

  private def processLDA(modelInit: PipelineModel, dataset: Dataset[Row], sparkSession: SparkSession) = {
    val lda = new LDA().setK(12).setMaxIter(60).setLearningDecay(0.12)
    val model = lda.fit(dataset)

    val topicsDataset: Dataset[Seq[String]] = describeTopicsWithPlainTextWords(modelInit, model)

    topicsDataset.show(false)
  }

  private def describeTopicsWithPlainTextWords(modelInit: PipelineModel, model: LDAModel) = {
    //     Describe topics.
    val topicIndices = model.describeTopics(7).map {
      s =>
        (s.getAs[Seq[Int]]("termIndices"), s.getAs[Seq[Double]]("termWeights"))
    }

    val vocab = modelInit.stages(2).asInstanceOf[CountVectorizerModel].vocabulary

    val topicsDataset = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map {
        case (term, _) => vocab(term.toInt)
      }
    }
    topicsDataset
  }

  private def prepareForModelling(dataInitial: Dataset[Row]) = {

    val regTokenizer = new org.apache.spark.ml.feature.RegexTokenizer()
      .setToLowercase(true)
      .setMinTokenLength(3)
      .setPattern("@")
      .setInputCol("finished_stem")
      .setOutputCol("tokenNew")

    val stopTokenizer = new StopWordsRemover()
      .setInputCol("tokenNew")
      .setOutputCol("stopToken")

    val countVectorizer = new CountVectorizer()
      .setInputCol("stopToken")
      .setOutputCol("features")

    val pipelineFinal = new Pipeline().setStages(Array(
      regTokenizer,
      stopTokenizer,
      countVectorizer))

    val modelInit : PipelineModel = pipelineFinal.fit(dataInitial)

    val dataset = modelInit
      .transform(dataInitial)
      .repartition(96, col("postId"))

    (modelInit, dataset)
  }
}
