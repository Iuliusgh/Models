package com.example.models.ui.theme

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectDetector(private val tflite: Interpreter, private val labels: List<String>) {
    data class Detection(val location: RectF, val label: String, val score: Float)
    private val ROW_SIZE = tflite.getOutputTensor(0).shape()[1]//4 coord numbers (x_center,y_center,width,height) + #classes
    private val CLASS_OFFSET = ROW_SIZE - labels.size

    private val outputBuffer = TensorBuffer.createFixedSize(tflite.getOutputTensor(0).shape(),tflite.getOutputTensor(0).dataType())
    lateinit var floatBuffer : IntArray

    private val detections : List<Detection> get() = (0 until tflite.getOutputTensor(0).shape()[2]).map{
        Detection(
            location = outputToRectF(it*ROW_SIZE),
            label = outputToLabel(it*ROW_SIZE+CLASS_OFFSET),
            score = outputToScore(it*ROW_SIZE+CLASS_OFFSET)
        )

    }
    private fun outputToRectF(index:Int): RectF {
        val x = floatBuffer[index].toFloat()
        val y = floatBuffer[index+1].toFloat()
        val w = floatBuffer[index+2].toFloat()
        val h = floatBuffer[index+3].toFloat()
        return RectF(x-w/2, y-h/2, x + w/2, y + h/2)
    }
    private fun outputToLabel(start: Int): String{
        val classConfValues = floatBuffer.copyOfRange(start,start+ROW_SIZE-CLASS_OFFSET)
        val maxIdx = classConfValues.withIndex().maxByOrNull(){it.value}?.index
        if (maxIdx != null) {
            return labels[maxIdx]
        }
        else return ""
    }
    private fun outputToScore(start: Int): Float{
        val classConfValues = floatBuffer.copyOfRange(start, start+ROW_SIZE-CLASS_OFFSET)
        val maxVal = classConfValues.maxOrNull()?.toFloat()
        return if (maxVal != null && maxVal > CONFIDENCE_THRESHOLD && maxVal < 1.0f) maxVal else 0.0f
    }
    fun detect(image: TensorImage): List<Detection> {
        tflite.run(image.buffer, outputBuffer.buffer)
        floatBuffer=outputBuffer.intArray
        return detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
        const val CONFIDENCE_THRESHOLD = 0.3f
    }
}