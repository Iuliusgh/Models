@file:OptIn(ExperimentalUnsignedTypes::class)

package com.example.models.ui.theme

import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectDetector(private val tflite: Interpreter, private val labels: List<String>) {
    data class Detection(val location: RectF, val label: String, val score: Float)
    private val ROW_SIZE = tflite.getOutputTensor(0).shape()[1]//4 coord numbers (x_center,y_center,width,height) + #classes
    private val CLASS_OFFSET = ROW_SIZE - labels.size

    private val outputBuffer = TensorBuffer.createFixedSize(tflite.getOutputTensor(0).shape(),DataType.UINT8)
    var floatBuffer : FloatArray = FloatArray(outputBuffer.flatSize)

    private val detections : List<Detection> get() = (0 until tflite.getOutputTensor(0).shape()[2]).map{
        Detection(
            location = outputToRectF(it*ROW_SIZE),
            label = outputToLabel(it*ROW_SIZE+CLASS_OFFSET),
            score = outputToScore(it*ROW_SIZE+CLASS_OFFSET)
        )

    }
    private fun outputToRectF(index:Int): RectF {
        val x = floatBuffer[index]
        val y = floatBuffer[index+1]
        val w = floatBuffer[index+2]
        val h = floatBuffer[index+3]
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
        val maxVal = classConfValues.maxOrNull()
        return if (maxVal != null && maxVal > CONFIDENCE_THRESHOLD && maxVal < 1.0f) maxVal.toFloat() else 0.0f
    }
    fun detect(image: TensorImage): List<Detection> {
        outputBuffer.buffer.rewind()
        tflite.run(image.buffer, outputBuffer.buffer)
        for (i in 0 until outputBuffer.flatSize) {
            floatBuffer[i] = outputBuffer.buffer[i]/127f
        }
        return detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
        const val CONFIDENCE_THRESHOLD = 0.3f
    }
}