import org.apache.avro.generic.GenericData
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.feature._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap

object SentimentTrainer {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    val twitterPath = args(0)

    val twitterData = readTwitterData(twitterPath, spark)

    val tokenizer = new RegexTokenizer()
      .setInputCol("Collapsed")
      .setOutputCol("Tokenized All")
      .setPattern("\\s+")

    val wordTokenizer = new RegexTokenizer()
      .setInputCol("Collapsed")
      .setOutputCol("Tokenized Words")
      .setPattern("\\W")

    val stopW = new StopWordsRemover()
      .setInputCol("Tokenized Words")
      .setOutputCol("Stopped")

    val w2v = new Word2Vec()
      .setInputCol("Stopped")
      .setOutputCol("features")

    val ngram = new NGram()
        .setN(2)
        .setInputCol("Stopped")
        .setOutputCol("Grams")

    val tokenVectorizer = new CountVectorizer()
      .setInputCol("Tokenized All")
      .setOutputCol("Token Vector")

    val gramVectorizer = new CountVectorizer()
      .setInputCol("Grams")
      .setOutputCol("Gram Vector")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Token Vector"))//, "Gram Vector"))
      .setOutputCol("features")


    val lr = new LogisticRegression()
      .setFamily("multinomial")
      .setLabelCol("Sentiment")


    val pipe = new Pipeline()
        .setStages(Array(tokenizer,
          wordTokenizer,
          stopW,
//          w2v,
          ngram,
          tokenVectorizer,
          gramVectorizer,
          assembler,
          lr
        ))

    val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(gramVectorizer.vocabSize, 10000)
//      .put(w2v.vectorSize, 300)
//      .put(w2v.maxIter, 20)
      .put(lr.elasticNetParam, .8)
      .put(lr.tol, 1e-20)
      .put(lr.maxIter, 100)

    val model = pipe.fit(twitterData, paramMap)

    val tr = model.transform(twitterData).select("Collapsed", "Sentiment", "probability", "prediction")
    tr.take(10).foreach(println)

    val eval = new BinaryClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setRawPredictionCol("prediction")

    val roc = eval.evaluate(tr)
    println(s"ROC: ${roc}")


//    tr.printSchema()

//    val paramGrid = new ParamGridBuilder()
//      .addGrid(tokenVectorizer.vocabSize, Array(10000))
//      .addGrid(gramVectorizer.vocabSize, Array(10000))
//      .addGrid(lr.elasticNetParam, Array(.8))
//      .addGrid(lr.tol, Array(1e-20))
//      .addGrid(lr.maxIter, Array(100))
//      .build()
//
//    val cv = new CrossValidator()
//      .setEstimator(pipe)
//      .setEvaluator(new BinaryClassificationEvaluator()
//        .setRawPredictionCol("prediction")
//        .setLabelCol("Sentiment"))
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(5)  // Use 3+ in practice
//      .setParallelism(1)
//
//    val model = cv.fit(twitterData)
//
//    model.transform(twitterData)
//      .select("ItemID","Collapsed", "probability", "prediction")
//      .collect().take(10)
//      .foreach(println)
//
//    println("Metrics: \n\n\n")
//    model.avgMetrics.foreach(println)
//    println("\n\n\n")
//
//    model.write.overwrite().save("sentiment-classifier")

  }

  def stripPunctuation(twit: String)= {
    twit.trim.replaceAll("""([\p{Punct}])\s*""", "")
  }

  def readTwitterData(path: String, spark: SparkSession) = {

    val schema = StructType("ItemID Sentiment SentimentText"
      .split(" ")
      .map(fieldName => {
        if (fieldName == "ItemID" || fieldName == "Sentiment")
          StructField(fieldName, IntegerType, nullable = false)
        else
          StructField(fieldName, StringType, nullable = false)
      }))

    val data = spark.read.format("csv")
      .schema(schema)
      .option("header", "true")
      .load(path)

    val collapse: String => String = {
      _.trim
        .replaceAll("([.,!?])", " $1")
        .replaceAll("((.))\\1+","$1")
    }
    val collapseUDF = udf(collapse)

    val newCol = collapseUDF.apply(data("SentimentText"))
    data.withColumn("Collapsed", newCol)
      .select("ItemID","Sentiment","Collapsed")

  }
}


