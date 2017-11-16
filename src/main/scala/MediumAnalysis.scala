import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.annotators.{Normalizer, RegexTokenizer}
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{NGram, StopWordsRemover}
import org.apache.spark.sql.SparkSession

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
    spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "xxx")
    spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "xxx")
    spark.sparkContext.hadoopConfiguration.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

    spark
  }

  def main(args: Array[String]): Unit = {
    val sparkSession = getSparkSession("Medium")

    val data = sparkSession.sqlContext
        .read
        .json("s3n://test-datasc-redshift-temp/medium-posts.json")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val ngram = new NGram()
      .setN(2)
      .setInputCol("words")
      .setOutputCol("ngrams")

    val remover = new StopWordsRemover()
      .setInputCol("ngrams")
      .setOutputCol("filtered")

    val normalizer = new Normalizer()
      .setInputCols(Array("filtered"))
      .setOutputCol("normalized")

    val finisher = new Finisher()
      .setInputCols("token")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        ngram,
        remover,
        normalizer,
        finisher
      ))

    pipeline
      .fit(data)
      .transform(data)
      .show()
  }
}
