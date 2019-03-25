package tech.thomaslegrand.kotlinml

import krangl.*
import smile.classification.LogisticRegression
import smile.validation.Validation.cv

fun main() {
    val iris = DataFrame.readCSV("src/main/resources/iris.csv")

    println(iris.schema())

    val y = iris["species"].toIntArray()
    val X = iris.remove("species")

    val logit = LogisticRegression(X.toDoubleMatrix().transpose(), y, 0.0, 0.001, 500)

    val test = arrayOf(3.1, 2.9, 1.7, 4.1)
    println(logit.predict(test.toDoubleArray()))
}
