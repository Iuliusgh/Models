package com.example.models.ui.theme

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectDetector(private val tflite: Interpreter, private val labels: List<String>) {
    data class Detection(val location: RectF, val label: String, val score: Float)

    val outputBuffer = TensorBuffer.createFixedSize(tflite.getOutputTensor(0).shape(),tflite.getOutputTensor(0).dataType())

    //output of yolo inference [1,84,8400] -> [8400,84]
    private val transposedOutput = Array(tflite.getOutputTensor(0).shape()[1]){FloatArray(tflite.getOutputTensor(0).shape()[2])}

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
    val detections get() = (0 until MAX_DETECTIONS).map{
        /*Detection(
            location = outputToRectF(yoloOutput[0][0][it],
                             yoloOutput[0][1][it],
                             yoloOutput[0][2][it],
                             yoloOutput[0][3][it]),
            label = outputToLabel(yoloOutput[0][it]),
            score = outputToScore(yoloOutput[0][it])
        )*/

    }
    private fun outputToRectF(x:Float, y: Float, w: Float, h: Float): RectF {
        return RectF(x-w/2, y+h/2, x + w/2, y - h/2)
    }
    private fun outputToLabel(outputArray: FloatArray): String{
        val classConfValues = outputArray
        val maxVal = classConfValues.maxOrNull()
        val maxIdx = classConfValues.indexOfFirst { it == maxVal }
        return labels[maxIdx]


    }
    private fun outputToScore(outputArray: FloatArray): Float{
        val classConfValues = outputArray.copyOfRange(4, outputArray.size)
        val maxVal = classConfValues.maxOrNull()
        return maxVal?:0f
    }
    fun detect(image: TensorImage): List<Detection>? {
        tflite.run(image.buffer, outputBuffer.buffer)
        val outputArray = outputBuffer.floatArray
        for(i in 0 until tflite.getOutputTensor(0).shape()[2]){
            for(j in 0 until tflite.getOutputTensor(0).shape()[1]) {
                transposedOutput[j][i] = outputArray[i * j + j]
            }
        }
        return null//detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
    }
}