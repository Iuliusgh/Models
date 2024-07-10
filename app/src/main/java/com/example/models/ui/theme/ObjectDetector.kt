package com.example.models.ui.theme

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectDetector(private val tflite: Interpreter, private val labels: List<String>) {
    data class Detection(val location: RectF, val label: String, val score: Float)
    var ROW_SIZE = tflite.getOutputTensor(0).shape()[1]//4 coord numbers (x_center,y_center,width,height) + #classes

    val outputBuffer = TensorBuffer.createFixedSize(tflite.getOutputTensor(0).shape(),tflite.getOutputTensor(0).dataType())

    /*
    Location tensor (kTfLiteFloat32):
    tensor of size [1 x num_results x 4], the inner array representing bounding boxes in the form [top, left, right, bottom].
    BoundingBoxProperties are required to be attached to the metadata and must specify type=BOUNDARIES and coordinate_type=RATIO.
     */
    private val locations = arrayOf(Array(MAX_DETECTIONS){FloatArray(4)})
    /*
    Classes tensor (kTfLiteFloat32):
    tensor of size [1 x num_results], each value representing the integer index of a class.
    if label maps are attached to the metadata as TENSOR_VALUE_LABELS associated files, they are used to convert the tensor values into labels.
     */
    private val classes =  arrayOf(FloatArray(MAX_DETECTIONS))
    /*
    scores tensor (kTfLiteFloat32):
    tensor of size [1 x num_results], each value representing the score of the detected object.
     */
    private val scores =  arrayOf(FloatArray(MAX_DETECTIONS))
    /*
    Number of detection tensor (kTfLiteFloat32):
    integer num_results as a tensor of size [1]. 
     */
    private val numResults = FloatArray(1)
    private val detections : List<Detection> get() = (0 until MAX_DETECTIONS).map{
        Detection(
            location = outputToRectF(it*ROW_SIZE),
            label = outputToLabel(it*ROW_SIZE),
            score = outputToScore(it*ROW_SIZE)
        )

    }
    private fun outputToRectF(index:Int): RectF {
        val x = outputBuffer.floatArray[index]
        val y = outputBuffer.floatArray[index+1]
        val w = outputBuffer.floatArray[index+2]
        val h = outputBuffer.floatArray[index+3]
        return RectF(x-w/2, y+h/2, x + w/2, y - h/2)
    }
    private fun outputToLabel(start: Int): String{
        val classConfValues = outputBuffer.floatArray.copyOfRange(start,start+ROW_SIZE-4)
        val maxVal = classConfValues.maxOrNull()
        if (maxVal != null && maxVal > 0) {
            val maxIdx = classConfValues.indexOfFirst { it == maxVal }
            return labels[maxIdx]
        }
        else return ""
    }
    private fun outputToScore(start: Int): Float{
        val classConfValues = outputBuffer.floatArray.copyOfRange(start, start+ROW_SIZE-4)
        val maxVal = classConfValues.maxOrNull()
        return maxVal?:0f
    }
    fun detect(image: TensorImage): List<Detection> {
        tflite.run(image.buffer, outputBuffer.buffer)
        return detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
    }
}