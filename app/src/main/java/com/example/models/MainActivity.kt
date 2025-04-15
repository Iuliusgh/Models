package com.example.models

import com.example.models.Model

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.BatteryManager
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.models.databinding.ActivityMainBinding
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
import kotlin.time.Duration
import kotlin.time.measureTime


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private val TAG = "Main"
    private lateinit var activityMainBinding: ActivityMainBinding
    private val model = YOLO(this)
    private val interpreter = Interpreter(this)
    private var modelPath =""
    private val datasetPath = "/storage/emulated/0/Dataset/coco/val2017"
    private var isModelSelected = false
    private var isDeviceSelected = false
    private val placeholder = listOf("---")
    private val preTime = Array(5000) { Duration.ZERO }
    private val runTime = Array(5000) { Duration.ZERO }
    private val postTime = Array(5000) { Duration.ZERO }
    private val energyConsumption = Array(5000) { 0 }

    //private var tfliteInterpreter : Interpreter? = null
    /*private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }*/

    private var selectedDevice = -1
    private var selectedModel = "No model"
    //TFLITE aux variables
    private lateinit var TFLITEInputBuffer : ByteBuffer
    private lateinit var TFLITEOutputBuffer: ByteBuffer
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        initUI()
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
        if (parent.getItemAtPosition(position) != placeholder[0]) {
            when (parent.id) {
                activityMainBinding.modelSelector.id -> {
                    activityMainBinding.selectedModel.text = parent.getItemAtPosition(position).toString()
                    if (interpreter.isInitialized()) {
                        interpreter.close()
                    }
                    try {
                        modelPath = "tflite/$parent.getItemAtPosition(position).toString()"
                        isModelSelected=true
                        model.loadModelFile(modelPath)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error loading model\n$e")
                    }
                }
                activityMainBinding.device.id -> {
                    selectedDevice = position - 1
                    try{
                        interpreter.initializeOptions(position)
                        //activityMainBinding.datatypeText.text = dataType.toString()
                        isDeviceSelected=true
                    }
                    catch(e:Exception){
                        Toast.makeText(this,e.toString(),Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
        if(isModelSelected && isDeviceSelected){
            activityMainBinding.button.isEnabled=true
        }
    }
    override fun onNothingSelected(parent: AdapterView<*>?) {}
    private fun initUI(){
        //Button to start the validation
        activityMainBinding.button.setOnClickListener {
            CoroutineScope(Dispatchers.Default).launch {
                interpreter.initializeInterpreter(model)
                validate()
            }
            activityMainBinding.button.isEnabled = false
        }
        //Fill the spinner with the available interpreters
        val modelDir = placeholder + assets.list("tflite")!!.toList()//File(,"tflite").listFiles()//!!.map{it.name}
        val modelArray = ArrayAdapter(this,android.R.layout.simple_spinner_item,modelDir)
        modelArray.setDropDownViewResource(android.R.layout.simple_spinner_item)
        activityMainBinding.modelSelector.adapter=modelArray
        activityMainBinding.device.adapter=ArrayAdapter(this,android.R.layout.simple_spinner_item,placeholder + interpreter.getDeviceList())
        //ArrayAdapter.createFromResource(this,R.array.available_interpreters, android.R.layout.simple_spinner_item).also {
            //it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            //activityMainBinding.interpreterSelector.adapter = it
        //}
        activityMainBinding.modelSelector.onItemSelectedListener = this
        activityMainBinding.device.onItemSelectedListener = this
        //activityMainBinding.modelSelector.setSelection(-1)
        //activityMainBinding.device.setSelection(-1)
    }

    @SuppressLint("DefaultLocale")
    private suspend fun validate() {
        val datasetChunk = 5000
        val json: MutableList<String> = mutableListOf()
        //val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val imgList = File(datasetPath).list()?.sorted()
        var info: String
        val output: MutableList<FloatArray> = MutableList(300) { FloatArray(6) }
        var outputSize : Int
        var tik : Int
        var tok : Int
        var pre: Duration
        var run : Duration
        var post : Duration
        Log.i(TAG,"Starting benchmark...")
        if (imgList != null) {
            for ((i, file) in imgList.take(datasetChunk).withIndex()) {
                pre = measureTime {
                    model.preprocess()
                    "$datasetPath/$file"
                    if(interpreter.isInputQuantized()){
                        quantize(model.getModelInput(),interpreter.getInputQuant())
                        //for (index in inputBuffer.indices){
                            //inputBuffer[index] = inputBuffer[index] / inputQuant.scale + inputQuant.zeroPoint
                        //}
                    }
                    array2Buffer(TFLITEInputBuffer,model.getModelInput(),interpreter.getInputDatatype())
                    //TFLITEInputBuffer.put(inputBuffer)//TensorBuffer.loadArray() ya clampea a [0,255]
                }
                run = measureTime {
                    //tik = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                    interpreter.run(TFLITEInputBuffer, TFLITEOutputBuffer)
                    //tok = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

                }
                post = measureTime {
                    TFLITEOutputBuffer.rewind()
                    buffer2Array(TFLITEOutputBuffer,modelOutput,interpreter.getOutputDatatype())
                    if(interpreter.isOutputQuantized()){
                        dequantize(modelOutput,interpreter.getOutputQuant())
                        //for (index in outputBuffer.indices){
                            //outputBuffer[index] = (outputBuffer[index] - outputQuant.zeroPoint) * outputQuant.scale
                        //}
                    }
                    outputSize=nms(modelOutput, output, outputShape)
                    postprocess(output, outputSize, preprocessResult.ratio, preprocessResult.pad, preprocessResult.ogImgShape)
                }
                TFLITEInputBuffer.clear()
                TFLITEOutputBuffer.clear()
                if(outputSize!=0){
                    json.add(outputToJSON(output, file,outputSize))
                }
                energyConsumption[i] = tok-tik
                preTime[i]=pre
                //activityMainBinding.preTimeVal.text=pre.toString()
                runTime[i]=run
                //activityMainBinding.runTimeVal.text=run.toString()
                postTime[i]=post
               //activityMainBinding.postTimeVal.text=post.toString()
                info = (i.toFloat() / datasetChunk * 100f).toString()
                withContext(Dispatchers.Main) {
                    activityMainBinding.progress.text = "Progress: $info %"
                    //activityMainBinding.energyVal.text= "${energyConsumption[i]} mAh"
                }
            }
        }
        json[0] = "["+json[0]
        json[json.size - 1]=json[json.size - 1]+"]"
        writeOutput(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),selectedModel + "_" + devices[selectedDevice]!!.replace(" ","_") + "_" + "output.json", json)
        val preTimeVal = preTime.reduce { acc, duration -> acc + duration }/datasetChunk
        val runTimeVal = runTime.reduce { acc, duration -> acc + duration }/datasetChunk
        val postTimeVal = postTime.reduce { acc, duration -> acc + duration }/datasetChunk
        val energyVal = (energyConsumption.reduce { acc, l -> acc + l }).toFloat()/datasetChunk
        withContext(Dispatchers.Main) {
            activityMainBinding.progress.text = "Completed"
            activityMainBinding.preTimeVal.text="AVG: ${preTimeVal}"
            activityMainBinding.runTimeVal.text="AVG: ${runTimeVal}"
            activityMainBinding.postTimeVal.text="AVG: ${postTimeVal}"
            activityMainBinding.energyVal.text="AVG: ${energyVal} uAh"
            activityMainBinding.button.isEnabled = true
        }
    }
    private fun writeOutput(path : File, name :String, json:MutableList<String>){
        val dirName = "InferenceResult"
        val newPath =File(path,dirName)
        if (!File(path,dirName).exists()){
            newPath.mkdirs()
        }
        val text = json.joinToString(separator = ",")
        val file  = File(newPath,name)
        try {
            if(file.exists()){
                file.delete()
            }
            //file.createNewFile()
            Log.i("FileWrite","Writing output to file...")
            file.writeText(text)
            if(file.readText()!=text){
                throw Exception("Error writing output to file, text mismatch")
            }
            //file.outputStream().write(text.toByteArray(Charsets.UTF_8))
            //file.outputStream().fd.sync()
            //file.outputStream().close()
        }
        catch (e:Exception) {
            Log.e("FileWrite", "Output to file failed" + e.toString())
            throw(e)
        }
        Log.i("FileWrite","Output written to file successfully.")
    }
    public override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }


}
