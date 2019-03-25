package tech.thomaslegrand.kotlinml

import krangl.*
import smile.projection.PCA
import smile.classification.LogisticRegression

fun main() {
    val iris = DataFrame.readCSV("src/main/resources/iris.csv")

    val y = iris["species"].toIntArray()
    val X = iris.remove("species")

    val pca = PCA(X.toDoubleMatrix())
    val logit = LogisticRegression(X.toDoubleMatrix().transpose(), y)
}
