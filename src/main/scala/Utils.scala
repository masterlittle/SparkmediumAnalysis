import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.annotators.{Lemmatizer, Normalizer, RegexTokenizer, Stemmer}
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by grofers on 16/12/17.
  */
object Utils {

  def preProcessTextData(data: Dataset[Row]): Dataset[Row] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("content")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("normalized"))
      .setOutputCol("lemma")
      .setLemmaDict("e_lemma.txt")

    val stemmer = new Stemmer()
      .setInputCols(Array("lemma"))
      .setOutputCol("stem")

    val finisher = new Finisher()
      .setInputCols("stem")
      .setCleanAnnotations(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        normalizer,
        lemmatizer,
        stemmer,
        finisher
      ))

    val dataInitial = pipeline
      .fit(data)
      .transform(data)
      .repartition(96, col("postId"))
    dataInitial
  }
}
