package tech.thomaslegrand.kotlinml

import krangl.DataCol
import krangl.Factor
import krangl.asFactor
import krangl.asType
import smile.math.matrix.JMatrix

// Helper function to transpose a Double Array
fun Array<out DoubleArray>.transpose(): Array<out DoubleArray>? = JMatrix(this).transpose().array()

// Helper function to convert single column to Int Array
fun DataCol.toIntArray(): IntArray = this.asFactor().asType<Factor>().map { factor -> factor?.index!! }.toIntArray()