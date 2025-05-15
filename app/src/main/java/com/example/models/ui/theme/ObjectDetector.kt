@file:OptIn(ExperimentalUnsignedTypes::class)

package com.example.models.ui.theme

/*

class ObjectDetector(private val yolov11n: Interpreter, private val labels: List<String>) {

    private val ROW_SIZE = yolov11n.getOutputTensor(0).shape()[1]//4 coord numbers (x_center,y_center,width,height) + #classes
    private val CLASS_OFFSET = ROW_SIZE - labels.size
    private  var maxConfIndex : Int = -1

    private val outputBuffer = TensorBuffer.createFixedSize(yolov11n.getOutputTensor(0).shape(),yolov11n.getOutputTensor(0).dataType())
    lateinit var floatBuffer : FloatArray

    private val detections : List<Detection> get() = (0 until yolov11n.getOutputTensor(0).shape()[2]).map{
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
        yolov11n.run(image.buffer, outputBuffer.buffer)
        floatBuffer= outputBuffer.floatArray.map { it/255f }.toFloatArray()
        //val aux =outputBuffer.buffer
        return detections
    }
    companion object {
        const val MAX_DETECTIONS = 10
        const val CONFIDENCE_THRESHOLD = 0.3f
    }
}*/