@file:OptIn(ExperimentalUnsignedTypes::class)

package com.example.models.ui.theme

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.vision.detector.Detection


class ObjectDetector(private val tflite: Interpreter, private val labels: List<String>) {

    private val ROW_SIZE = tflite.getOutputTensor(0).shape()[1]//4 coord numbers (x_center,y_center,width,height) + #classes
    private val CLASS_OFFSET = ROW_SIZE - labels.size
    private  var maxConfIndex : Int = -1

    private val outputBuffer = TensorBuffer.createFixedSize(tflite.getOutputTensor(0).shape(),tflite.getOutputTensor(0).dataType())
    lateinit var floatBuffer : FloatArray

    private val detections : List<Detection> get() = (0 until tflite.getOutputTensor(0).shape()[2]).map{
        Detection.create(
            outputToRectF(it*ROW_SIZE),
            listOf(Category(indexToLabel(),outputToScore(it*ROW_SIZE+CLASS_OFFSET)))
        )
    }
    private fun outputToRectF(index:Int): RectF {
        val x = floatBuffer[index]
        val y = floatBuffer[index+1]
        val w = floatBuffer[index+2]
        val h = floatBuffer[index+3]
        return RectF(x-w/2, y-h/2, x + w/2, y + h/2)
    }
    private fun indexToLabel(): String{
        return if(maxConfIndex>0)labels[maxConfIndex] else ""
    }
    private fun outputToScore(start: Int): Float{
        val classConfValues = floatBuffer.copyOfRange(start, start+ROW_SIZE-CLASS_OFFSET)
        maxConfIndex = classConfValues.withIndex().maxByOrNull { it.value }?.index ?: -1
        return if (maxConfIndex>0)classConfValues[maxConfIndex] else 0f
    }
    fun detect(image: TensorImage): List<Detection> {
        tflite.run(image.buffer, outputBuffer.buffer)
        floatBuffer= outputBuffer.floatArray
        for(i in floatBuffer.indices){
            floatBuffer[i] /= 255f
        }
        return detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
        const val CONFIDENCE_THRESHOLD = 0.3f
    }
}