import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by grofers on 14/12/17.
  */
object MediumClassifier {

  val scrapedTimestamp = 1510941734

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

    val data = prepareRawData(sparkSession)


    val dataPerUnitTime = data
      .withColumn("diffTime", lit(scrapedTimestamp) - col("publishedAt") / 1000)
      .withColumn("clapsPerUnitTime", col("claps")./(col("diffTime")))
      .withColumn("responsePerUnitTime", col("responseCreated") / col("diffTime"))
      .withColumn("usersPerUnitTime", col("userCount") / col("diffTime"))


    val countColumnLength: (String) => Int = (sentence: String) => {
      sentence.split(' ').length
    }


    //Declare the UDF
    val countColumnLengthUDF = udf(countColumnLength)

    val moreData = dataPerUnitTime.withColumn("titleCount", countColumnLengthUDF(dataPerUnitTime.col("title")))

    doProcessing(sparkSession, moreData)
  }

  private def splitData(dataPerUnitTime: DataFrame, seed: Int) = {
    val split: Array[DataFrame] = dataPerUnitTime.randomSplit(Array(0.75, 0.25), seed)
    val training = split(0).cache()
    val testData = split(1).cache()
    (training, testData)
  }

  private def doProcessing(sparkSession: SparkSession, dataFrame: DataFrame) = {
    val preProcessedData: DataFrame = preProcessData(sparkSession, Utils.preProcessTextData(dataFrame))

    val (trainingData: DataFrame, testData: DataFrame) = splitData(preProcessedData, 5)

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("isPopular")
      .setOutputCol("indexedLabel")
      .fit(preProcessedData)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(preProcessedData)

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(15)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model.
    val model = pipeline.fit(trainingData)

    evaluateModel(testData, model)
  }

  private def evaluateModel(testOutput: DataFrame, model: PipelineModel) = {
    // Make predictions.
    val predictions = model.transform(testOutput)

    // Select (prediction, true label) and compute test error.

    val evaluator3 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val precision = evaluator3.evaluate(predictions)
    println("Precision score is = " + precision)

    val evaluator4 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val recall = evaluator4.evaluate(predictions)
    println("Recall score is = " + recall)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy) * 100)

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(rfModel.featureImportances)
  }

  private def preProcessData(sparkSession: SparkSession, data: DataFrame) = {

    val regTokenizer = new org.apache.spark.ml.feature.RegexTokenizer()
      .setToLowercase(true)
      .setMinTokenLength(3)
      .setPattern("@")
      .setInputCol("finished_stem")
      .setOutputCol("tokenNew")

    val stopTokenizer = new StopWordsRemover()
      .setInputCol("tokenNew")
      .setOutputCol("stopToken")

    val hashingTF = new HashingTF()
      .setInputCol("stopToken")
      .setOutputCol("rawFeatures")

    val featurizedData = new Pipeline().setStages(Array(
      regTokenizer,
      stopTokenizer,
      hashingTF))
      .fit(data)
      .transform(data)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("wordFeatures")
      .setMinDocFreq(10)

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "imageCount",
        "wordCount",
        "readingTime",
        "titleCount",
      "wordFeatures"))
      .setOutputCol("features")

    val trainingPipeline = new Pipeline()
      .setStages(Array(
        idf,
        assembler
      ))
      .fit(featurizedData)

    trainingPipeline.transform(featurizedData)
  }

  private def prepareRawData(sparkSession: SparkSession): DataFrame = {
    import sparkSession.sqlContext.implicits._

    val dataPopAndNotPop = sparkSession.sqlContext
      .read
      .json("s3n://test/medium-posts.json")

    val dataPopular = sparkSession.sqlContext
      .read
      .json("s3n://test/medium-posts-popular.json")

    // I first take take the union and then select only the non popular articles.
    // Then I union with the popular articles.
    // This is done because a direct union is not possible as there are some popular articles in the all articles dataset.

    val dataJoin = dataPopAndNotPop.union(dataPopular).select("postId").distinct()

    val dataNotPopular = dataJoin
      .except(dataPopular.select("postId"))
      .join(dataPopAndNotPop, "postId")
      .withColumn("isPopular", lit(0))

    val dataAll = dataNotPopular.union(
      dataPopular.withColumn("isPopular", lit(1)).select(dataNotPopular.columns.head, dataNotPopular.columns.tail: _*)
    )
      .withColumn("claps", col("claps").cast(IntegerType))
      .withColumn("imageCount", col("imageCount").cast(IntegerType))
      .withColumn("claps", when($"claps".isNull, 0).otherwise($"claps"))
      .withColumn("imageCount", when($"imageCount".isNull, 0.0).otherwise(abs($"imageCount".cast(DoubleType))))
    dataAll
  }
}
