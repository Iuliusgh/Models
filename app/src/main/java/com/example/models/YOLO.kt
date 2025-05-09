package com.example.models

import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.round

class YOLO(context: Context) : Model(context) {
    private val coco80to91: Map<Int, Int> = mapOf(
        0 to 1,    // person
        1 to 2,    // bicycle
        2 to 3,    // car
        3 to 4,    // motorcycle
        4 to 5,    // airplane
        5 to 6,    // bus
        6 to 7,    // train
        7 to 8,    // truck
        8 to 9,    // boat
        9 to 10,   // traffic light
        10 to 11,  // fire hydrant
        11 to 13,  // stop sign
        12 to 14,  // parking meter
        13 to 15,  // bench
        14 to 16,  // bird
        15 to 17,  // cat
        16 to 18,  // dog
        17 to 19,  // horse
        18 to 20,  // sheep
        19 to 21,  // cow
        20 to 22,  // elephant
        21 to 23,  // bear
        22 to 24,  // zebra
        23 to 25,  // giraffe
        24 to 27,  // backpack
        25 to 28,  // umbrella
        26 to 31,  // handbag
        27 to 32,  // tie
        28 to 33,  // suitcase
        29 to 34,  // frisbee
        30 to 35,  // skis
        31 to 36,  // snowboard
        32 to 37,  // sports ball
        33 to 38,  // kite
        34 to 39,  // baseball bat
        35 to 40,  // baseball glove
        36 to 41,  // skateboard
        37 to 42,  // surfboard
        38 to 43,  // tennis racket
        39 to 44,  // bottle
        40 to 46,  // wine glass
        41 to 47,  // cup
        42 to 48,  // fork
        43 to 49,  // knife
        44 to 50,  // spoon
        45 to 51,  // bowl
        46 to 52,  // banana
        47 to 53,  // apple
        48 to 54,  // sandwich
        49 to 55,  // orange
        50 to 56,  // broccoli
        51 to 57,  // carrot
        52 to 58,  // hot dog
        53 to 59,  // pizza
        54 to 60,  // donut
        55 to 61,  // cake
        56 to 62,  // chair
        57 to 63,  // couch
        58 to 64,  // potted plant
        59 to 65,  // bed
        60 to 67,  // dining table
        61 to 70,  // toilet
        62 to 72,  // tv
        63 to 73,  // laptop
        64 to 74,  // mouse
        65 to 75,  // remote
        66 to 76,  // keyboard
        67 to 77,  // cell phone
        68 to 78,  // microwave
        69 to 79,  // oven
        70 to 80,  // toaster
        71 to 81,  // sink
        72 to 82,  // refrigerator
        73 to 84,  // book
        74 to 85,  // clock
        75 to 86,  // vase
        76 to 87,  // scissors
        77 to 88,  // teddy bear
        78 to 89,  // hair drier
        79 to 90   // toothbrush
    )
    override val datasetPath: String = super.datasetPath + "coco/val2017"
    override val exportFileExtension = ".json"
    private val iou = 0.7f
    private val confidence = 0.001f
    private val inputSize:Int by lazy {inputShape[1]}
    private val padFillValue = Scalar(114.0, 114.0, 114.0, 0.0)
    private lateinit var resizeRatio: Pair<Float, Float>
    private lateinit var resizePad: Pair<Int, Int>//left,top
    private lateinit var originalImgShape: org.opencv.core.Size
    private var nmsMaxCandidates  = 30000 //maximum number of detection candidates to apply nms to
    private var nmsMaxDetections = 300 //maximum number of final detections
    private var maxWH = 7860 // maximum width/height of the image
    private val nmsReshapedInput  by lazy { Array(1) { Array(outputShape[2]) { FloatArray(outputShape[1]) }}}//1,8400,84
    private var outputSize = 0
    private val detections = MutableList(nmsMaxDetections) { FloatArray(6) }
    private val detectionsJSON:MutableList<String> = mutableListOf()
    private var inputFilename: String = ""

    override fun <String>preprocess(imgPath:String) {
        inputFilename = imgPath.toString().split("/").last()
        val mat = Imgcodecs.imread(imgPath.toString(), Imgcodecs.IMREAD_COLOR)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB)
        originalImgShape = mat.size()
        val r = minOf(inputSize / originalImgShape.width, inputSize / originalImgShape.height).toFloat()
        resizeRatio = Pair(r, r)
        val newShape = org.opencv.core.Size(round(originalImgShape.width * r), round(originalImgShape.height * r))
        val wpad = (inputSize - newShape.width) / 2
        val hpad = (inputSize - newShape.height) / 2
        val top = round(hpad - 0.1).toInt()
        val bottom = round(hpad + 0.1).toInt()
        val left = round(wpad - 0.1).toInt()
        val right = round(wpad + 0.1).toInt()
        resizePad=Pair(left,top)
        val paddedImg = Mat()
        val floatImg = Mat()
        if (originalImgShape!=newShape) {
            val resizedImg = Mat()
            Imgproc.resize(mat, resizedImg, newShape, 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.copyMakeBorder(resizedImg, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, padFillValue)
        }
        else {
            Core.copyMakeBorder(mat, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, padFillValue)
        }
        paddedImg.convertTo(floatImg, CvType.CV_32FC3)
        Core.divide(floatImg,Scalar(255.0,255.0,255.0),floatImg)
        floatImg.get(0, 0, modelInput)
    }
    override suspend fun postprocess() {
        nms()
        scaleBoxesToImage()
    }
    override fun inferenceOutputToExportFormat() {
        if (outputSize>0){
            detectionsJSON.add(detection2JSON(inputFilename))
        }
    }
    override fun serializeResults(): String {
        return formatJSONForExport()
    }

    private suspend fun nms() {
        //Variable declaration

        //1D array to [1][8400][84]
       parallelArrayOperation(modelOutput.size,{ i ->
            if(i<outputShape[2]*4) {
                nmsReshapedInput[0][i % outputShape[2]][(i / outputShape[2]) % outputShape[1]] = modelOutput[i] * inputSize.toFloat()
            }
            else{
                nmsReshapedInput[0][i % outputShape[2]][(i / outputShape[2]) % outputShape[1]] = modelOutput[i]
            }
        })
        //# classes output shape - 4 bbox values
        val numClasses = outputShape[1] - 4
        //Unused
        //val numMasks = inferenceOutput.shape[1] - 4 - numClasses
        val candidates = Array(outputShape[2]) { false }
        parallelArrayOperation(outputShape[2],{ i ->
            var max = 0f
            for (j in 4 until outputShape[1]) {
                val value = nmsReshapedInput[0][i][j]
                if (value > max)
                    max = value
            }
            candidates[i] = max > confidence
            xywh2xyxy(nmsReshapedInput[0][i])
        })
        val x: ArrayList<FloatArray> = ArrayList()
        for (i in candidates.indices) {
            if (candidates[i])
                x.add(nmsReshapedInput[0][i])
        }
        var keep:List<Int> = emptyList()
        if (x.isNotEmpty()) {
            val box: Array<FloatArray> = Array(x.size) { FloatArray(4) }
            val clas: Array<FloatArray> = Array(x.size) { FloatArray(numClasses) }
            for (i in x.indices) {
                box[i] = x[i].copyOfRange(0, 4)
                clas[i] = x[i].copyOfRange(4, 4 + numClasses)
            }
            val i: ArrayList<Int> = ArrayList()
            val j: ArrayList<Int> = ArrayList()
            for (a in clas.indices) {
                for (b in clas[a].indices) {
                    if (clas[a][b] > confidence) {
                        i.add(a)
                        j.add(b)
                    }
                }
            }
            val preds = MutableList(i.size) { box[i[it]] + x[i[it]][4 + j[it]] + j[it].toFloat()}
            preds.sortByDescending { it[4] }
            val offset = FloatArray(preds.size) { a -> preds[a][5]*maxWH }
            val numDetections = preds.size
            if (numDetections > 0) {
                if (numDetections > nmsMaxCandidates) {
                    preds.subList(nmsMaxCandidates, numDetections).clear()
                }
                val nmsCand = Array(preds.size) { FloatArray(6) }
                for (index in preds.indices) {
                    nmsCand[index][0] = preds[index][0] + offset[index]
                    nmsCand[index][1] = preds[index][1] + offset[index]
                    nmsCand[index][2] = preds[index][2] + offset[index]
                    nmsCand[index][3] = preds[index][3] + offset[index]
                    nmsCand[index][4] = preds[index][4]
                    nmsCand[index][5] = preds[index][5]
                }
                keep = nonMaxSuppression(nmsCand).take(nmsMaxDetections)
                for (index in keep.indices) {
                    detections[index] = preds[keep[index]]
                }
            }
        }
        outputSize = keep.size
    }
    private fun nonMaxSuppression(candidates: Array<FloatArray>): List<Int> {
        val areas = candidates.map { (it[2] - it[0]) * (it[3] - it[1])  }
        val keep = ArrayList<Int>()
        for (i in candidates.indices) {
            var flag = true
            var j = 0
            while (j in keep.indices && flag) {
                val topLeftX = maxOf(candidates[i][0], candidates[keep[j]][0])
                val topLeftY = maxOf(candidates[i][1], candidates[keep[j]][1])
                val bottomRightX = minOf(candidates[i][2], candidates[keep[j]][2])
                val bottomRightY = minOf(candidates[i][3], candidates[keep[j]][3])
                var width = bottomRightX - topLeftX
                var height = bottomRightY - topLeftY
                if (width < 0) width = 0f
                if (height < 0) height = 0f
                val inter = width * height
                val union = areas[i] + areas[keep[j]] - inter
                if ((inter / union) > iou) {
                    flag = false
                }
                j++
            }
            if (flag) {
                keep.add(i)
            }
        }
        return keep
    }
    private fun scaleBoxesToImage() {
        val left = resizePad.first.toFloat()
        val top = resizePad.second.toFloat()
        val width = originalImgShape.width.toFloat()
        val height = originalImgShape.height.toFloat()
        val divVal = minOf(resizeRatio.first, resizeRatio.second)
        for (i in 0 until outputSize) {
            detections[i][0] = ((detections[i][0] - left) / divVal).coerceIn(0f, width)
            detections[i][1] = ((detections[i][1] - top) / divVal).coerceIn(0f, height)
            detections[i][2] = ((detections[i][2] - left) / divVal).coerceIn(0f, width) - detections[i][0]
            detections[i][3] = ((detections[i][3] - top) / divVal).coerceIn(0f, height) - detections[i][1]
            detections[i][4] = detections[i][4]
            detections[i][5] = detections[i][5]
        }
    }
    private fun xywh2xyxy(outputCoords: FloatArray) {
        val hw = outputCoords[2] / 2
        val hh = outputCoords[3] / 2
        outputCoords[2] = outputCoords[0] + hw
        outputCoords[3] = outputCoords[1] + hh
        outputCoords[0] -= hw
        outputCoords[1] -= hh
    }

    private fun detection2JSON(filename: String): String {
        return buildString {
            for (i in 0 until outputSize) {
                append(String.format(
                    """{"image_id":${filename.substringBeforeLast(".").toInt()},""" +
                            """"category_id":${coco80to91[detections[i][5].toInt()]},""" +
                            """"bbox":[${round(detections[i][0]*1e3).toInt()/1e3f},${round(detections[i][1]*1e3).toInt()/1e3f},${round(detections[i][2]*1e3).toInt()/1e3f},${round(detections[i][3]*1e3).toInt()/1e3f}],""" +
                            """"score":${round(detections[i][4]*1e6).toInt()/1e6f}}"""
                ))
                 if (i != outputSize - 1) {
                    append(",")
                }
            }
        }
    }
    private fun formatJSONForExport():String{
        detectionsJSON[0] = "["+detectionsJSON[0]
        detectionsJSON[detectionsJSON.lastIndex] += "]"
        return detectionsJSON.joinToString(separator = ",")
    }
    override fun clearResultList() {
        detectionsJSON.clear()
    }

}