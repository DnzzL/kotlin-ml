package tech.thomaslegrand.kotlinml

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.StringIndexer


fun main() {
    val conf = SparkConf()
        .setMaster("local")
        .setAppName("Kotlin Spark")

    val sc = JavaSparkContext(conf)
    val spark = SparkSession
        .builder()
        .appName("Kotlin Spark")
        .orCreate

    val iris = spark.read()
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("src/main/resources/iris.csv")
    iris.show(5)
    iris.printSchema()

    val indexer = StringIndexer()
        .setInputCol("species")
        .setOutputCol("label")
        .fit(iris)
    val indexed = indexer.transform(iris)
    indexed.show(5)

    val assembler = VectorAssembler()
        .setInputCols(arrayOf("sepal_length", "sepal_width", "petal_length", "petal_width"))
        .setOutputCol("features")

    val pca = PCA()
        .setInputCol("features")
        .setOutputCol("pcaFeatures")
        .setK(2)

    val lr = LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.1)
        .setElasticNetParam(0.8)
        .setFeaturesCol(pca.outputCol)
        .setLabelCol("label")

    //creating pipeline
    val pipeline = Pipeline().setStages(arrayOf(assembler, pca, lr))

    val paramGrid = ParamGridBuilder()
        .addGrid(pca.k(), intArrayOf(2, 3))
        .addGrid(lr.regParam(), doubleArrayOf(0.1, 0.01))
        .build()

    val cv = CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(MulticlassClassificationEvaluator())
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setSeed(12)

    val cvModel = cv.fit(indexed)

    println(cvModel.avgMetrics().toList())
    cvModel.write().overwrite().save("logistic_regression.model")
}
