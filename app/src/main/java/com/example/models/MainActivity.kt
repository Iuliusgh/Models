package com.example.models

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.BatteryManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import androidx.appcompat.app.AppCompatActivity
import com.example.models.databinding.ActivityMainBinding
import com.qualcomm.qti.QnnDelegate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor
import java.io.File
import java.io.FileInputStream
import java.lang.Runtime.getRuntime
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.round
import kotlin.math.roundToInt
import kotlin.time.Duration
import kotlin.time.measureTime


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private val QNNTAG = "Qnn_Interpreter"
    private lateinit var activityMainBinding: ActivityMainBinding
    private val inputSize: Int = 224
    private val confidence = 0.001f
    private val iou = 0.7f
    private val modelPath =
        //"tflite/test_full_integer_quant_with_int16_act.tflite"
    //"tflite/densenet121-snapdragon_8_gen_2.tflite"
    // "tflite/inception_v3_quantized.tflite"
        //"tflite/test_float16.tflite"
        "tflite/test_integer_quant.tflite"

    private val datasetPath = "/storage/emulated/0/Dataset/coco/val2017"

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
    private val preTime = Array(5000) { Duration.ZERO }
    private val runTime = Array(5000) { Duration.ZERO }
    private val postTime = Array(5000) { Duration.ZERO }
    private val energyConsumption = Array(5000) { 0 }
    private lateinit var inputQuant : Tensor.QuantizationParams
    private lateinit var outputQuant : Tensor.QuantizationParams
    private lateinit var inputShape : IntArray
    private lateinit var outputShape : IntArray
    data class PreprocessResult(
        val ratio: Pair<Float, Float>,
        val pad: Pair<Int, Int>,
        val ogImgShape: org.opencv.core.Size
    )

    private lateinit var tfliteInterpreter : Interpreter
    /*private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }*/
    private lateinit var dataType: DataType
    private lateinit var inputBuffer : FloatArray
    private lateinit var outputBuffer : FloatArray
    //TFLITE aux variables
    private lateinit var TFLITEInputBuffer : ByteBuffer
    private lateinit var TFLITEOutputBuffer: ByteBuffer
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        //val pv = PlatformValidator(PlatformValidatorUtil.Runtime.DSP).runtimeCheck(application)
        initUI()
        //Permissions to access dataset
        if (checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_IMAGES), 0)
        }
        /*
        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        // Get current battery level in percentage
        val batteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        // Get current battery voltage (in mV)
        val batteryConsumption = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER) / 1e6
        // Get average current (in microamperes)
        val batteryCurrent = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE)
        // Get instantaneous current (in microamperes)
        val batteryCurrentNow = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

        println("Battery Level: $batteryLevel%")
        println("Remaining Energy: $batteryConsumption  mWh")
        println("Battery Current (Avg): $batteryCurrent µA")
        println("Battery Current (Now): $batteryCurrentNow µA")
        */
    }
    override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
        initializeInterpreter(parent.getItemAtPosition(position).toString())
    }
    override fun onNothingSelected(parent: AdapterView<*>?) {
        TODO("Not yet implemented")
    }
    private fun initUI(){
        //Button to start the validation
        activityMainBinding.button.setOnClickListener {
            initializeInterpreter("")
            CoroutineScope(Dispatchers.Default).launch {
                validate()
            }
            activityMainBinding.button.isEnabled = false
        }
        //Fill the spinner with the available interpreters
        //ArrayAdapter.createFromResource(this,R.array.available_interpreters, android.R.layout.simple_spinner_item).also {
            //it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            //activityMainBinding.interpreterSelector.adapter = it
        //}
        //activityMainBinding.interpreterSelector.onItemSelectedListener = this
    }
    private fun initializeInterpreter(selectedInterpreter :String){
        var text = ""
        val qnnDelegate = initNPUDelegate()
        //Log.i(QNNTAG,"QNNdelegate available: " + qnnDelegate.isAvailable.toString())
        val interpreterOptions = Interpreter.Options()
        interpreterOptions.runtime = InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY
        interpreterOptions.setAllowBufferHandleOutput(true)
        interpreterOptions.setUseNNAPI(false)
        interpreterOptions.numThreads = getRuntime().availableProcessors()
        //interpreterOptions.setUseXNNPACK(true)
        try {
            interpreterOptions.addDelegate(qnnDelegate)
        }
        catch (e:Exception){
            Log.e("Delegate","Delegate error" ,e)
        }
        //interpreterOptions.addDelegate(QnnDelegate(QnnDelegate.Options()))
        val modelFile = loadModelFileToBuffer()
        //modelFile.order(ByteOrder.nativeOrder())
        try {
            tfliteInterpreter = Interpreter(modelFile, interpreterOptions)
        }
        catch (e: Exception){
            Log.e("QNN_Int","Cannot initialize the qnn interpreter",e)
        }
        dataType = tfliteInterpreter.getInputTensor(0).dataType()
        inputQuant = tfliteInterpreter.getInputTensor(0).quantizationParams()
        outputQuant = tfliteInterpreter.getOutputTensor(0).quantizationParams()
        inputShape = tfliteInterpreter.getInputTensor(0).shape()
        outputShape = tfliteInterpreter.getOutputTensor(0).shape()
        TFLITEInputBuffer = ByteBuffer.allocateDirect(inputShape.reduce{acc, i -> acc * i } * tfliteInterpreter.getInputTensor(0).dataType().byteSize())
        //TFLITEInputBuffer = TensorBuffer.createFixedSize(tfliteInterpreter.getInputTensor(0).shape(), tfliteInterpreter.getInputTensor(0).dataType())
        TFLITEInputBuffer.order(ByteOrder.nativeOrder())
        TFLITEOutputBuffer = ByteBuffer.allocateDirect(outputShape.reduce{acc, i -> acc * i } * tfliteInterpreter.getOutputTensor(0).dataType().byteSize())
        //TFLITEOutputBuffer = TensorBuffer.createFixedSize(tfliteInterpreter.getOutputTensor(0).shape(), tfliteInterpreter.getOutputTensor(0).dataType())
        TFLITEOutputBuffer.order(ByteOrder.nativeOrder())
    /*"SNPE" -> {
        val model = resources.openRawResource(R.raw.test_float32_quantized)
        snpeNN = SNPE.NeuralNetworkBuilder(application)
            .setRuntimeOrder(NeuralNetwork.Runtime.DSP)
            .setModel(model,model.available())
            .build()

    }*/

        activityMainBinding.modelInfo.text=text

    }
    private fun loadModelFileToBuffer():MappedByteBuffer{
        val fd = assets.openFd(modelPath)
        val fileChannel = FileInputStream(fd.fileDescriptor).channel
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY,fd.startOffset,fd.declaredLength)
        return buffer
    }
    private fun initNPUDelegate(): Delegate {
        var text = ""
        val options = QnnDelegate.Options()
        //options.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_VERBOSE)
        options.skelLibraryDir = applicationInfo.nativeLibraryDir
        //options.libraryPath = applicationInfo.nativeLibraryDir
        options.cacheDir = cacheDir.absolutePath
        options.modelToken="7d94710092391108927ab0fe94cee635"
        if(QnnDelegate.checkCapability((QnnDelegate.Capability.DSP_RUNTIME))){
            options.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND)
            text = "Model executing on: DSP"
        }
        else{
            val isHTP16 = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
            val isHTPQuant = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)
            if(!isHTP16 && !isHTPQuant){
                Log.e(QNNTAG,"NPU not supported")
            }
            options.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            options.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            options.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            text = "Model executing on: HTP_Int"

            if (isHTP16){
            options.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            text = "Model executing on: HTP_Float16"
           }
        }
        //options.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16)
        //options.setHtpPdSession(QnnDelegate.Options.HtpPdSession.HTP_PD_SESSION_UNSIGNED)
        //options.setBackendType(QnnDelegate.Options.BackendType.UNDEFINED_BACKEND)
        return QnnDelegate(options)
    }
    @SuppressLint("DefaultLocale")
    private suspend fun validate() {
        var isIntModel = false
        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val imgList = File(datasetPath).list()?.sorted()
        var preprocessResult: PreprocessResult
        var info: String
        val output: MutableList<FloatArray> = MutableList(300) { FloatArray(6) }
        var outputSize : Int
        var tik = 0
        var tok = 0
        var pre: Duration
        var run : Duration
        var post : Duration
        inputBuffer = FloatArray(inputShape.reduce{acc, i -> acc * i } )
        outputBuffer = FloatArray(outputShape.reduce{acc, i -> acc * i } )
        if (imgList != null) {
            for ((i, file) in imgList.withIndex()) {
                pre = measureTime {
                    preprocessResult = preprocessImage("$datasetPath/$file",inputBuffer)
                    if(inputQuant.scale != 0.0f){
                        for (index in inputBuffer.indices){
                            inputBuffer[index] = inputBuffer[index] / inputQuant.scale + inputQuant.zeroPoint
                        }
                    }
                    array2Buffer(TFLITEInputBuffer,inputBuffer,dataType)
                    //TFLITEInputBuffer.put(inputBuffer)//TensorBuffer.loadArray() ya clampea a [0,255]
                }
                run = measureTime {
                    tik = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                    tfliteInterpreter.run(TFLITEInputBuffer, TFLITEOutputBuffer)
                    tok = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

                }
                post = measureTime {
                    TFLITEOutputBuffer.rewind()
                    buffer2Array(TFLITEOutputBuffer,outputBuffer,dataType)
                    if(outputQuant.scale != 0.0f){
                        for (index in outputBuffer.indices){
                            outputBuffer[index] = (outputBuffer[index] - outputQuant.zeroPoint) * outputQuant.scale
                        }
                    }
                    //outputSize=nms(outputBuffer, output, outputShape)
                    //postprocess(output, outputSize, preprocessResult.ratio, preprocessResult.pad, preprocessResult.ogImgShape)
                }
                TFLITEInputBuffer.clear()
                TFLITEOutputBuffer.clear()
                //if(outputSize!=0){
                    //json.add(outputToJSON(output, file,outputSize))
                //}
                energyConsumption[i] = tok-tik
                preTime[i]=pre
                runTime[i]=run
                postTime[i]=post
                info = (i.toFloat() / imgList.size.toFloat() * 100f).toString()
                withContext(Dispatchers.Main) {
                    activityMainBinding.progress.text = "Progress: $info %"
                    activityMainBinding.energyVal.text= "${energyConsumption[i]} mAh"
                }
            }
        }
        //json[0] = "["+json[0]
        //json[json.size - 1]=json[json.size - 1]+"]"
        File(filesDir, "output.json").writeText(json.joinToString(separator = ","))
        withContext(Dispatchers.Main) {
            activityMainBinding.progress.text = "Completed"
            activityMainBinding.preTimeVal.text="AVG: ${preTime.reduce { acc, duration -> acc + duration }/5000}"
            activityMainBinding.runTimeVal.text="AVG: ${runTime.reduce { acc, duration -> acc + duration }/5000}"
            activityMainBinding.postTimeVal.text="AVG: ${postTime.reduce { acc, duration -> acc + duration }/5000}"
            activityMainBinding.energyVal.text="AVG: ${(energyConsumption.reduce { acc, l -> acc + l }).toFloat()/5e3} uAh"
        }
    }
    public override fun onDestroy() {
        tfliteInterpreter.close()
        super.onDestroy()
    }
    private fun preprocessImage(imgPath:String, floatBuffer: FloatArray): PreprocessResult {
        val padFillValue = Scalar(114.0, 114.0, 114.0,0.0)
        val mat = Imgcodecs.imread(imgPath, Imgcodecs.IMREAD_COLOR)
        Imgproc.cvtColor(mat,mat,Imgproc.COLOR_BGR2RGB)
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
        val floatImg = Mat()
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
        paddedImg.convertTo(floatImg,CvType.CV_32F,1/255.0)
        /*if(isSNPEModel){
            val offset = floatImg.total().toInt()
            for(i in 0 until floatImg.rows()){
                for(j in 0 until floatImg.cols()){
                    val pixel = floatImg.get(i,j)
                    floatBuffer[i*j+j] = pixel[0].toFloat()
                    floatBuffer[i*j+j + offset] = pixel[1].toFloat()
                    floatBuffer[i*j+j + 2*offset] = pixel[2].toFloat()
                }
            }
        }
        else{*/
            floatImg.get(0,0,floatBuffer)
        //}

        return PreprocessResult(ratio, Pair(left, top), imgShape)
    }
    private suspend fun nms(inferenceOutput: FloatArray, output: MutableList<FloatArray>, outputShape: IntArray, maxNMS: Int = 30000, maxWH: Int = 7860, maxDet: Int = 300,  ): Int {
        //Variable declaration
        val outputSize = outputShape[1]
        val numOutputs = outputShape[2]
        val reshapedInput = Array(1) { Array(numOutputs) { FloatArray(outputSize) } } //1,8400,84
        //1D array to [1][84][8400]
        val numThreads = getRuntime().availableProcessors()
        var chunkSize = inferenceOutput.size / numThreads
        val jobs = mutableListOf<Job>()
        for (i in 0 until numThreads) {
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
        chunkSize = numOutputs / numThreads
        for (i in 0 until numThreads) {
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
    private fun xywh2xyxy(outputCoords: FloatArray) {
        val hw = outputCoords[2] / 2
        val hh = outputCoords[3] / 2
        outputCoords[2] = outputCoords[0] + hw
        outputCoords[3] = outputCoords[1] + hh
        outputCoords[0] -= hw
        outputCoords[1] -= hh
    }

    private fun nonMaxSuppression(
        candidates: Array<FloatArray>
    ): List<Int> {

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

private fun array2Buffer(buffer: ByteBuffer, array : FloatArray, bufferDataType : DataType) {
    for (i in array.indices) {
        val byteArray = ByteArray(bufferDataType.byteSize())
        for (j in 0 until bufferDataType.byteSize()){
            if(bufferDataType==DataType.FLOAT32){
                byteArray[j]=array[i].toRawBits().shr(j*8).toByte()
            }
            else{
                byteArray[j]=array[i].roundToInt().shr(j*8).toByte()
            }

        }
        buffer.put(byteArray)
    }
    buffer.rewind()
}
private fun buffer2Array(buffer: ByteBuffer, array: FloatArray, bufferDataType: DataType){
    val byteSize = bufferDataType.byteSize()
    for(i in 0 until buffer.capacity()/byteSize){
        var bits : Int = 0x00
        for (j in 0 until byteSize){
            bits =  bits or (buffer[i*byteSize+j].toInt() and 0xFF).shl(8*j)
        }
        array[i] = bits.toFloat()
    }
}

