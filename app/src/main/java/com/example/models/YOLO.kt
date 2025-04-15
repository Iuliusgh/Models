package com.example.models

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.round

class YOLO(context: Context) : Model<String>(context) {
    private val iou = 0.7f
    private val confidence = 0.001f
    private val inputSize = inputShape[1]
    private lateinit var resizeRatio: Pair<Float, Float>
    private lateinit var resizePad: Pair<Int, Int>//left,top
    private lateinit var originalImgShape: org.opencv.core.Size

    override fun preprocess(imgPath:String) {
        val padFillValue = Scalar(114.0, 114.0, 114.0, 0.0)
        val mat = Imgcodecs.imread(imgPath, Imgcodecs.IMREAD_COLOR)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB)
        originalImgShape = mat.size()
        val r = minOf(inputSize / originalImgShape.width, inputSize / originalImgShape.height).toFloat()
        resizeRatio = Pair(r, r)
        val newShape = org.opencv.core.Size(
            round(originalImgShape.width * r),
            round(originalImgShape.height * r)
        )
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
        paddedImg.convertTo(floatImg, CvType.CV_32F, 1 / 255.0)
        floatImg.get(0, 0, modelInput)
    }
    private suspend fun nms(inferenceOutput: FloatArray, output: MutableList<FloatArray>, outputShape: IntArray, maxNMS: Int = 30000, maxWH: Int = 7860, maxDet: Int = 300): Int {
        //Variable declaration
        val outputSize = outputShape[1]
        val numOutputs = outputShape[2]
        val reshapedInput = Array(1) { Array(numOutputs) { FloatArray(outputSize) } } //1,8400,84
        //1D array to [1][84][8400]
        var chunkSize = inferenceOutput.size / 8
        val jobs = mutableListOf<Job>()
        for (i in 0 until 8) {
            val start = chunkSize * i
            val end = chunkSize * (i + 1)
            val job = CoroutineScope(Dispatchers.Default).launch {
                for (index in start until end) {
                    if(index<numOutputs*4) {
                        reshapedInput[0][index % numOutputs][(index / numOutputs) % outputSize] = inferenceOutput[index] * inputSize
                    }
                    else{
                        reshapedInput[0][index % numOutputs][(index / numOutputs) % outputSize] = inferenceOutput[index]
                    }
                }
            }
            jobs.add(job)
        }
        for (job in jobs) {
            job.join()
        }
        jobs.clear()
        //# classes output shape - 4 bbox values
        val numClasses = outputSize - 4
        //Unused
        //val numMasks = inferenceOutput.shape[1] - 4 - numClasses
        val candidates = Array(numOutputs) { false }
        chunkSize = numOutputs / 8
        for (i in 0 until 8) {
            val start = chunkSize * i
            val end = chunkSize * (i + 1)
            val job = CoroutineScope(Dispatchers.Default).launch {
                for (index in start until end) {
                    var max = 0f
                    for (j in 4 until outputSize) {
                        val value = reshapedInput[0][index][j]
                        if (value > max)
                            max = value
                    }
                    candidates[index] = max > confidence
                    xywh2xyxy(reshapedInput[0][index])
                }
            }
            jobs.add(job)
        }
        for (job in jobs) {
            job.join()
        }
        jobs.clear()
        val x: ArrayList<FloatArray> = ArrayList()
        for (i in candidates.indices) {
            if (candidates[i])
                x.add(reshapedInput[0][i])
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
                if (numDetections > maxNMS) {
                    preds.subList(maxNMS, numDetections).clear()
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
                keep = nonMaxSuppression(nmsCand).take(maxDet)
                for (index in keep.indices) {
                    output[index] = preds[keep[index]]
                }
            }
        }
        return keep.size
    }
    fun postprocess(inferenceOutput: List<FloatArray>, outputSize: Int, ratio: Pair<Float, Float>, pad: Pair<Int, Int>, ogImgShape: org.opencv.core.Size, ) {
        val left = pad.first.toFloat()
        val top = pad.second.toFloat()
        val width = ogImgShape.width.toFloat()
        val height = ogImgShape.height.toFloat()
        val divVal = minOf(ratio.first, ratio.second)
        for (i in 0 until outputSize) {
            inferenceOutput[i][0] = ((inferenceOutput[i][0] - left) / divVal).coerceIn(0f, width)
            inferenceOutput[i][1] = ((inferenceOutput[i][1] - top) / divVal).coerceIn(0f, height)
            inferenceOutput[i][2] = ((inferenceOutput[i][2] - left) / divVal).coerceIn(0f, width) - inferenceOutput[i][0]
            inferenceOutput[i][3] = ((inferenceOutput[i][3] - top) / divVal).coerceIn(0f, height) - inferenceOutput[i][1]
            inferenceOutput[i][4] = inferenceOutput[i][4]
            inferenceOutput[i][5] = inferenceOutput[i][5]
        }
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
    private fun xywh2xyxy(outputCoords: FloatArray) {
        val hw = outputCoords[2] / 2
        val hh = outputCoords[3] / 2
        outputCoords[2] = outputCoords[0] + hw
        outputCoords[3] = outputCoords[1] + hh
        outputCoords[0] -= hw
        outputCoords[1] -= hh
    }
    private fun coco80to91(eightyClass:Int):Int{
        val coco80to91: Map<Int, Int> = mapOf(
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
        return coco80to91[eightyClass]!!
    }
    fun outputToJSON(output: List<FloatArray>, filename: String, outputSize: Int): String {
        return buildString {
            for (i in 0 until outputSize) {
                append(String.format(
                    """{"image_id":${filename.substringBeforeLast(".").toInt()},""" +
                            """"category_id":${coco80to91(output[i][5].toInt())},""" +
                            """"bbox":[${round(output[i][0]*1e3).toInt()/1e3f},${round(output[i][1]*1e3).toInt()/1e3f},${round(output[i][2]*1e3).toInt()/1e3f},${round(output[i][3]*1e3).toInt()/1e3f}],""" +
                            """"score":${round(output[i][4]*1e6).toInt()/1e6f}}"""
                ))
                if (i != outputSize - 1) {
                    append(",")
                }
            }
        }
    }
}