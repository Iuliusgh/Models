package com.example.models

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.models.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil.loadMappedFile
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import kotlin.math.round
import kotlin.time.Duration
import kotlin.time.measureTime


class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding

    private val inputSize: Int = 640
    private val confidence = 0.001f
    private val iou = 0.7f
    private val modelPath = "test_float32.tflite"

    //private val labelPath = "coco80_labels.txt"
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
    private val json: MutableList<String> = mutableListOf()

    //NMS Candidate object
    //data class Candidate(val x: Float, val y: Float, val w: Float, val h: Float, var score: Float, var clas: Int)
    data class PreprocessResult(
        val ratio: Pair<Float, Float>,
        val pad: Pair<Int, Int>,
        val ogImgShape: org.opencv.core.Size
    )


    private val tfImageProcessor by lazy {
        //val cropSize = minOf(inputSize, inputSize)
        ImageProcessor.Builder().add(CastOp(DataType.FLOAT32)).add(NormalizeOp(0f, 255f)).build()
        //.add(ResizeWithCropOrPadOp(cropSize, cropSize)).add(ResizeOp(tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.BILINEAR))
    }
    private val tflite by lazy {
        Interpreter(
            loadMappedFile(this, modelPath), Interpreter.Options().apply {
                this.addDelegate(GpuDelegate(GpuDelegateFactory.Options()))
            }
        )
    }

    /*private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }*/

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        activityMainBinding.button.setOnClickListener {
            CoroutineScope(Dispatchers.Default).launch {
                validate()
            }
            activityMainBinding.button.isEnabled = false
        }
    }

    @SuppressLint("DefaultLocale")
    private suspend fun validate() {
        val path = "dataset/coco/val2017"
        val imgList = assets.list(path)//File(filesDir, "/dataset/coco/val2017/")
        val outputBuffer = TensorBuffer.createFixedSize(
            tflite.getOutputTensor(0).shape(), tflite.getOutputTensor(0).dataType()
        )
        var bitmap: Bitmap
        val preProBitmap: Bitmap =
            Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val tensorImage = TensorImage(DataType.FLOAT32)
        val floatBuffer = FloatArray(inputSize * inputSize * 3)
        var preprocessResult: PreprocessResult
        var info: String
        val mat = Mat()
        val output: MutableList<FloatArray> = MutableList(30000) { FloatArray(6) }
        var outputSize : Int
        var preTime: Duration
        var runTime: Duration
        var postTime: Duration
        var preTimeSum: Duration = Duration.ZERO
        var runTimeSum: Duration = Duration.ZERO
        var postTimeSum: Duration = Duration.ZERO
        if (imgList != null) {
            for ((i, file) in imgList.withIndex()) {
                preTime= measureTime {
                    assets.open("$path/$file").use { inputStream -> bitmap = BitmapFactory.decodeStream(inputStream) }
                    Utils.bitmapToMat(bitmap, mat)
                    preprocessResult = preprocessImage(mat, preProBitmap,floatBuffer)
                    tensorImage.load(floatBuffer, intArrayOf(inputSize,inputSize,3))
                }
                preTimeSum += preTime
                withContext(Dispatchers.Main) {
                    activityMainBinding.preTimeVal.text = preTime.toString()
                }
                runTime = measureTime {
                    tflite.run(tensorImage.buffer, outputBuffer.buffer)
                }
                runTimeSum += runTime
                withContext(Dispatchers.Main) {
                    activityMainBinding.runTimeVal.text = runTime.toString()
                }
                postTime = measureTime {
                    outputSize=nms(outputBuffer, output)
                    postprocess(output, outputSize, preprocessResult.ratio, preprocessResult.pad, preprocessResult.ogImgShape)
                }
                postTimeSum += postTime
                withContext(Dispatchers.Main) {
                    activityMainBinding.postTimeVal.text = postTime.toString()
                }
                if(outputSize!=0){
                    json.add(outputToJSON(output, file,outputSize))
                }
                info = "Progress: " + (i / imgList.size.toFloat() * 100).toInt().toString() + "%"
                withContext(Dispatchers.Main) {
                    activityMainBinding.progress.text = info
                }
            }
        }
        json[0] = "["+json[0]
        json[json.size - 1]=json[json.size - 1]+"]"
        File(filesDir, "output.json").writeText(json.joinToString(separator = ","))
        withContext(Dispatchers.Main) {
            activityMainBinding.progress.text = "Completed"
            activityMainBinding.preTimeVal.text="AVG: ${preTimeSum/5000}"
            activityMainBinding.runTimeVal.text="AVG: ${runTimeSum/5000}"
            activityMainBinding.postTimeVal.text="AVG: ${postTimeSum/5000}"
        }
    }

    public override fun onDestroy() {
        tflite.close()
        super.onDestroy()
    }

    private fun preprocessImage(mat: Mat, bitmap: Bitmap, floatBuffer: FloatArray): PreprocessResult {
        val padFillValue = Scalar(114.0, 114.0, 114.0,0.0)
        val imgShape = mat.size()
        val r = minOf(inputSize / imgShape.width, inputSize / imgShape.height).toFloat()
        val ratio = Pair(r, r)
        val newShape = org.opencv.core.Size(round(imgShape.width * r), round(imgShape.height * r))
        val wpad = (inputSize - newShape.width) / 2
        val hpad = (inputSize - newShape.height) / 2
        val top = round(hpad - 0.1).toInt()
        val bottom = round(hpad + 0.1).toInt()
        val left = round(wpad - 0.1).toInt()
        val right = round(wpad + 0.1).toInt()
        val paddedImg = Mat()
        val normalizedImg = Mat()
        if (!imgShape.equals(newShape)) {
            val resizedImg = Mat()
            Imgproc.resize(mat, resizedImg, newShape, 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.copyMakeBorder(
                resizedImg, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, padFillValue
            )
        } else {
            Core.copyMakeBorder(
                mat, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, padFillValue
            )
        }
        paddedImg.convertTo(normalizedImg,CvType.CV_32F,1.0/255.0)
        Imgproc.cvtColor(normalizedImg,normalizedImg,Imgproc.COLOR_BGR2RGB)
        normalizedImg.get(0,0,floatBuffer)
        return PreprocessResult(ratio, Pair(top, left), imgShape)
    }

    private suspend fun nms(
        inferenceOutput: TensorBuffer,
        output: MutableList<FloatArray>,
        maxNMS: Int = 30000,
        maxWH: Int = 7860,
        maxDet: Int = 300,
    ): Int {
        //Variable declaration
        val floatArray = inferenceOutput.floatArray
        val outputShape = inferenceOutput.shape[1]//84-> classes + 4 bbox coords
        val numOutputs = inferenceOutput.shape[2]//8400 outputs
        val reshapedInput = Array(1) { Array(numOutputs) { FloatArray(outputShape) } } //1,8400,84
        //1D array to [1][84][8400]
        val numThreads = Runtime.getRuntime().availableProcessors()
        var chunkSize = inferenceOutput.flatSize / numThreads
        val jobs = mutableListOf<Job>()
        for (i in 0 until numThreads) {
            val start = chunkSize * i
            val end = chunkSize * (i + 1)
            val job = CoroutineScope(Dispatchers.Default).launch {
                for (index in start until end) {
                    reshapedInput[0][index % numOutputs][(index / numOutputs) % outputShape] = floatArray[index]
                }
            }
            jobs.add(job)
        }
        for (job in jobs) {
            job.join()
        }
        jobs.clear()
        //# classes output shape - 4 bbox values
        val numClasses = outputShape - 4
        //Unused
        //val numMasks = inferenceOutput.shape[1] - 4 - numClasses
        chunkSize = numOutputs / numThreads
        val candidates = Array(numOutputs) { false }
        for (i in 0 until numThreads) {
            val start = chunkSize * i
            val end = chunkSize * (i + 1)
            val job = CoroutineScope(Dispatchers.Default).launch {
                for (index in start until end) {
                    var max = 0f
                    for (j in 4 until outputShape) {
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
            val preds: ArrayList<FloatArray> = ArrayList()
            for (i in x.indices) {
                box[i][0] = x[i][0]*inputSize
                box[i][1] = x[i][1]*inputSize
                box[i][2] = x[i][2]*inputSize
                box[i][3] = x[i][3]*inputSize
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
            for (a in i.indices) {
                preds.add(box[i[a]] + x[i[a]][4 + j[a]] + j[a].toFloat())
            }
            val numDetections = preds.size
            if (numDetections > 0) {
                if (numDetections > maxNMS) {
                    preds.sortByDescending { it[4] }
                    preds.subList(maxNMS, numDetections).clear()
                }
                val nmsCand = Array(preds.size) { FloatArray(6) }
                for (index in preds.indices) {
                    nmsCand[index] = preds[index].copyOfRange(0, 6)
                }
                 keep = nonMaxSuppression(nmsCand)
                for (index in 0 until keep.size.coerceAtMost(maxDet)) {
                    output[index] = nmsCand[keep[index]]
                }
            }
        }
        return keep.size
    }

    private fun xywh2xyxy(outputCoords: FloatArray) {
        val hw = outputCoords[2] * 0.5f
        val hh = outputCoords[3] * 0.5f
        outputCoords[2] = outputCoords[0] + hw
        outputCoords[3] = outputCoords[1] + hh
        outputCoords[0] -= hw
        outputCoords[1] -= hh

    }

    private fun nonMaxSuppression(
        candidates: Array<FloatArray>
    ): List<Int> {

        candidates.sortByDescending { it[4] }//sort by score
        val areas = candidates.map { (it[2] - it[0]) * (it[3] - it[1]) }
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

    private fun postprocess(
        inferenceOutput: List<FloatArray>,
        outputSize: Int,
        ratio: Pair<Float, Float>,
        pad: Pair<Int, Int>,
        ogImgShape: org.opencv.core.Size,
    ) {
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

    @SuppressLint("DefaultLocale")
    fun outputToJSON(output: List<FloatArray>, filename: String, outputSize: Int): String {
        return buildString {
        for (i in 0 until outputSize) {
            append(String.format(
                """{"image_id":${filename.substringBeforeLast(".").toInt()},""" +
                        """"category_id":${coco80to91[output[i][5].toInt()]},""" +
                        """"bbox":[${output[i][0]},${output[i][1]},${output[i][2]},${output[i][3]}],""" +
                        """"score":${output[i][4]}}"""
            ))
            if (i != outputSize - 1) {
                append(",")
            }
        }
    }
    }
}

