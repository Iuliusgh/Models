package com.example.models

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import com.example.models.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil.loadMappedFile
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.round


class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding

    private val inputSize: Int = 640
    private val confidence = 0.001f
    private val iou = 0.7f
    private val modelPath = "test_float32.tflite"
    private val labelPath = "coco80_labels.txt"
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
    //NMS Candidate object
    data class Candidate(val bbox: FloatArray, val score: Float, val clas: Int)

    private val tfImageProcessor by lazy {
        val cropSize = minOf(inputSize, inputSize)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
    }
    private val nnApiDelegate by lazy {
        NnApiDelegate()
    }

    private val tflite by lazy {
        Interpreter(
            loadMappedFile(this, modelPath),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
        //Interpreter.Options().apply { addDelegate(FlexDelegate()) })
    }

    /*private val detector by lazy {
        ObjectDetector(
            tflite,
            FileUtil.loadLabels(this, labelPath)
        )
    }*/
    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        val path = "dataset/coco/val2017"
        val imgList = assets.list(path)//File(filesDir, "/dataset/coco/val2017/")

        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        // Start a timer or use a frame callback to update FPS

        val outputBuffer = TensorBuffer.createFixedSize(
            tflite.getOutputTensor(0).shape(),
            tflite.getOutputTensor(0).dataType()
        )

        if (imgList != null) {
            for (file in imgList) {
                val stream = assets.open(path + "/" + file)
                val bitmap = BitmapFactory.decodeStream(stream)
                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)
                val preprocessedImg = preprocessImage(mat)
                val tensorImage = TensorImage(DataType.FLOAT32)
                val noAlpha = Mat()
                Imgproc.cvtColor(preprocessedImg, noAlpha, Imgproc.COLOR_RGBA2RGB)
                val bmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(preprocessedImg, bmp)
                tensorImage.load(bmp)
                val inputImage = tfImageProcessor.process(tensorImage)
                tflite.run(inputImage.buffer, outputBuffer.buffer)
                postprocess(outputBuffer)
            }
        }

        activityMainBinding.textView.text = ""
    }

    public override fun onDestroy() {
        tflite.close()
        nnApiDelegate.close()
        super.onDestroy()
    }

    private fun preprocessImage(mat: Mat): Mat {
        val padFillValue = Scalar(114.0, 114.0, 114.0)
        val inputSize = 640
        val imgShape = mat.size()
        val r = minOf(inputSize / imgShape.width, inputSize / imgShape.height)
        val ratio = Pair(r, r)
        val newShape = org.opencv.core.Size(round(imgShape.width * r), round(imgShape.height * r))
        val wpad = (inputSize - newShape.width) / 2
        val hpad = (inputSize - newShape.height) / 2
        val top = round(hpad - 0.1).toInt()
        val bottom = round(hpad + 0.1).toInt()
        val left = round(wpad - 0.1).toInt()
        val right = round(wpad + 0.1).toInt()
        val paddedImg = Mat()
        if (!imgShape.equals(newShape)) {
            val resizedImg = Mat()
            Imgproc.resize(mat, resizedImg, newShape, 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.copyMakeBorder(
                resizedImg,
                paddedImg,
                top,
                bottom,
                left,
                right,
                Core.BORDER_CONSTANT,
                padFillValue
            )
        } else {
            Core.copyMakeBorder(
                mat,
                paddedImg,
                top,
                bottom,
                left,
                right,
                Core.BORDER_CONSTANT,
                padFillValue
            )
        }
        return paddedImg
    }

    private fun postprocess(
        inferenceOutput: TensorBuffer,
        maxNMS: Int = 30000,
        maxWH: Int = 7860,
        maxDet: Int = 300
    ) {

        val floatArray = inferenceOutput.floatArray
        val reshapedInput = Array(inferenceOutput.shape[0]) { //1
            Array(inferenceOutput.shape[1]) {//84
                FloatArray(inferenceOutput.shape[2])//8400
            }
        }
        val shuffledArray = Array(inferenceOutput.shape[0]) { //1
            Array(inferenceOutput.shape[2]) {//8400
                FloatArray(inferenceOutput.shape[1])//84
            }
        }
        for (i in floatArray.indices) {
            reshapedInput[0][i / inferenceOutput.shape[2] % inferenceOutput.shape[1]][i % inferenceOutput.shape[2]] =
                floatArray[i]
        }
        val numClasses = inferenceOutput.shape[1] - 4
        val numMasks = inferenceOutput.shape[1] - 4 - numClasses

        for (i in 0 until inferenceOutput.shape[1] * inferenceOutput.shape[2]) {//[1][84][8400] -> [1][8400][84]
            shuffledArray[0][i % inferenceOutput.shape[2]][i / inferenceOutput.shape[2]] =
                reshapedInput[0][i / inferenceOutput.shape[2]][i % inferenceOutput.shape[2]]
        }
        val candidates = shuffledArray[0].map { row ->
            (row.drop(4).max()) > confidence
        }
        //xywh2xyxy(shuffledArray[0])
        val x :List<FloatArray> =  candidates.mapIndexedNotNull { index, b ->
            if (b) shuffledArray[0][index] else null
        }
        if (x.size > 0) {
            val box: ArrayList<FloatArray> = ArrayList()
            val clas: ArrayList<FloatArray> = ArrayList()
            val preds: ArrayList<FloatArray> = ArrayList()
            for (i in x.indices) {
                box.add(x[i].copyOfRange(0, 4))
                clas.add(x[i].copyOfRange(4, 4 + numClasses))
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
                val nms_cand = ArrayList<Candidate>()

                val offset = FloatArray(numDetections)
                for (index in preds.indices) {
                    offset[index] = preds[index][5] * maxWH
                    nms_cand.add(
                        Candidate(
                            preds[index].copyOfRange(0, 4).map { it + offset[index] }.toFloatArray(),
                            preds[index][4],
                            preds[index][5].toInt()
                        )
                    )
                }


                val keep = nms(nms_cand)
                val result = keep.map { nms_cand[it] }
                val b = 1
            }

        }

        val a = 1
    }

    private fun xywh2xyxy(outputCoords: Array<FloatArray>) {
        for (i in outputCoords.indices) {
            val centerX = outputCoords[i][0]
            val centerY = outputCoords[i][1]
            val w = outputCoords[i][2]
            val h = outputCoords[i][3]
            outputCoords[i][0] = centerX - w
            outputCoords[i][1] = centerY - h
            outputCoords[i][2] = centerX + w
            outputCoords[i][3] = centerY + h
        }
    }

    private fun nms(
      candidates: ArrayList<Candidate>
    ): List<Int> {

        candidates.sortByDescending { it.score }
        val areas = candidates.map { it.bbox[2] * it.bbox[3] }
        val keep = ArrayList<Int>()
        for (i in candidates.indices) {
            var flag = true
            for (j in keep.indices) {
                val x = maxOf(
                    candidates[i].bbox[0] - candidates[i].bbox[2]/2,
                    candidates[keep[j]].bbox[0] - candidates[keep[j]].bbox[2]/2
                )
                val y = maxOf(
                    candidates[i].bbox[1] - candidates[i].bbox[3]/2,
                    candidates[keep[j]].bbox[1] - candidates[keep[j]].bbox[3]/2
                )
                var width = minOf(x + candidates[i].bbox[2], x + candidates[keep[j]].bbox[2]) - x
                var height = minOf(y + candidates[i].bbox[3], y + candidates[keep[j]].bbox[3]) - y
                if (width < 0) width = 0f
                if (height < 0) height = 0f
                val inter = width * height
                val union = areas[i] + areas[keep[j]] - inter
                if ((inter / union) > iou) {
                    flag = false
                }
            }
            if (flag) {
                keep.add(i)
            }
        }
        return keep
    }
}


