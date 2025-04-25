package com.example.models

import android.Manifest
import android.content.pm.PackageManager
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import kotlin.time.Duration
import kotlin.time.measureTime


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private val TAG = "Main"
    private lateinit var activityMainBinding: ActivityMainBinding
    private val model = ResNet(this)
    private val interpreter = Interpreter(this)
    private var modelPath =""
    private val datasetPath = "/storage/emulated/0/Dataset/coco/val2017"
    private var isModelSelected = false
    private var isDeviceSelected = false
    private val placeholder = listOf("---")
    private val preTime = Array(5000) { Duration.ZERO }
    private val runTime = LongArray(5000)
    private val postTime = Array(5000) { Duration.ZERO }
    private val energyConsumption = Array(5000) { 0 }
    //private var tfliteInterpreter : Interpreter? = null
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
                        modelPath = "tflite/${parent.getItemAtPosition(position)}"
                        isModelSelected=true
                        model.loadModelFile(modelPath)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error loading model\n$e")
                    }
                }
                activityMainBinding.device.id -> {
                    interpreter.selectExecutionDevice(position - 1)
                    try{
                        interpreter.initializeOptions()
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
                model.initializeIO()
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
    }
    private suspend fun validate() {
        val datasetChunk = 10
        //val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val imgList = File(datasetPath).list()?.sorted()
        var info = 0.0
        var tik : Int
        var tok : Int
        var pre: Duration
        var post : Duration
        val outputDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        Log.i(TAG,"Starting benchmark...")
        //if (imgList != null) {
            //for ((i, filename) in imgList.take(datasetChunk).withIndex()) {
                pre = measureTime {
                    model.preprocess(assets."dog.jpg")//"$datasetPath/$filename")
                    if(interpreter.isInputQuantized()){
                        quantize(model.modelInput,interpreter.getInputQuant())
                    }
                    array2Buffer(model.modelInput,interpreter.getInputBuffer(),interpreter.getInputDatatype())

                }
                //tik = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                interpreter.run()
                //tok = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                post = measureTime {
                    buffer2Array(interpreter.getOutputBuffer(),model.modelOutput,interpreter.getOutputDatatype())
                    if(interpreter.isOutputQuantized()){
                        dequantize(model.modelOutput,interpreter.getOutputQuant())
                    }
                    model.postprocess()
                }
                interpreter.clearIOBuffers()
                //model.inferenceOutputToExportFormat(filename)
                //energyConsumption[i] = tok-tik
                //preTime[i]=pre
                //activityMainBinding.preTimeVal.text=pre.toString()
                //runTime[i] = interpreter.getInferenceTimeNanoseconds()
                //activityMainBinding.runTimeVal.text=run.toString()
                //postTime[i]=post
               //activityMainBinding.postTimeVal.text=post.toString()
                //info = (i.toFloat() / datasetChunk * 1e2)
                withContext(Dispatchers.Main) {
                    activityMainBinding.progress.text = "Progress: $info %"
                    //activityMainBinding.energyVal.text= "${energyConsumption[i]} mAh"
                }
            //}
        //}
        writeOutputToFile(outputDir,"${modelPath.split("/").last().split(".").first()}_${interpreter.executingDevice}_${model.exportFileExtension}",model.serializeResults())
        val preTimeVal = preTime.reduce { acc, duration -> acc + duration }/datasetChunk
        val runTimeVal = runTime.reduce { acc, duration -> acc + duration }/(datasetChunk*1e6)
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
    public override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
    fun writeOutputToFile(path : File, filename :String, content: String){
        val dirName = "InferenceResult"
        val newPath = File(path,dirName)
        if (!File(path,dirName).exists()){
            newPath.mkdirs()
        }
        val file  = File(newPath,filename)
        try {
            if(file.exists()){
                file.delete()
            }
            //file.createNewFile()
            Log.i("FileWrite","Writing output to file...")
            file.writeText(content)
            if(file.readText()!=content){
                throw Exception("Error writing output to file, text mismatch")
            }
            //file.outputStream().write(text.toByteArray(Charsets.UTF_8))
            //file.outputStream().fd.sync()
            //file.outputStream().close()
        }
        catch (e:Exception) {
            Log.e("FileWrite", "Output to file failed $e")
            throw(e)
        }
        Log.i("FileWrite","Output written to file successfully.")
    }

}
