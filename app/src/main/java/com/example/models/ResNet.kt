package com.example.models

import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.round

class ResNet(context:Context): Model(context) {
    private lateinit var originalImgShape: Size
    private val resizeSize = 256
    private val cropSize = 224
    private val classificationResultList:MutableList<IntArray> = mutableListOf()
    override val datasetPath: String = super.datasetPath + "Imagenet/archive"
    override val exportFileExtension: String = ".csv"
    private val k = 5 // # of top results to take

    override fun <String> preprocess(imgPath:String) {
        val mat = Imgcodecs.imread(imgPath.toString(), Imgcodecs.IMREAD_COLOR)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB)
        originalImgShape = mat.size()
        val r = maxOf(resizeSize / originalImgShape.width, resizeSize / originalImgShape.height).toFloat()
        val newShape = Size(round(originalImgShape.width * r), round(originalImgShape.height * r))
        val resizedImg = Mat()
        Imgproc.resize(mat, resizedImg, newShape, 0.0, 0.0, Imgproc.INTER_LINEAR)
        val cropX = ((resizedImg.size().width - cropSize) / 2).toInt()
        val cropY = ((resizedImg.size().height - cropSize) / 2).toInt()
        val croppedImg = Mat(resizedImg, Rect(cropX,cropY,cropSize,cropSize))
        val floatImg = Mat()
        croppedImg.convertTo(floatImg, CvType.CV_32FC3)
        Core.divide(floatImg, Scalar(255.0,255.0,255.0),floatImg)
        Core.subtract(floatImg,Scalar(0.485, 0.456, 0.406),floatImg)
        Core.divide(floatImg, Scalar(0.229, 0.224, 0.225),floatImg)
        floatImg.get(0,0,modelInput)
    }


    override fun serializeResults(): String {
        return classificationResultList.joinToString(separator = ";"){ array -> array.joinToString(separator = ",")}
    }

    override fun inferenceOutputToExportFormat() {
        classificationResultList.add(modelOutput.withIndex().sortedByDescending { it.value }.take(k).map { it.index }.toIntArray())
    }

    override fun clearResultList() {
        classificationResultList.clear()
    }
}