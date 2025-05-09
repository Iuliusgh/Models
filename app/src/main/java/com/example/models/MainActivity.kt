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
import kotlin.math.floor
import kotlin.time.Duration
import kotlin.time.measureTime


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private val TAG = "Main"
    private lateinit var activityMainBinding: ActivityMainBinding
    private val model = ResNet(this)
    private val interpreter = Interpreter(this)
    private var modelPath =""
    private var isModelSelected = false
    private var isDeviceSelected = false
    private val placeholder = listOf("---")
    private val preTime:MutableList<Duration>  = mutableListOf()
    private val runTime:MutableList<Long>  = mutableListOf()
    private val postTime:MutableList<Duration>  = mutableListOf()
    private val energyConsumption = Array(5000) { 0 }
    private val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        if (checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_IMAGES), 0)
        }
        initUI()
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
            activityMainBinding.progress.text = getString(R.string.starting)
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
        //val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        Log.i(TAG,"Listing validation dataset...")
        val imgList = File(model.datasetPath).walkTopDown().filter{it.extension == "jpg"}.map { it.absolutePath }.toList().sorted()
        Log.i(TAG,"Finished listing.")
        val datasetChunk = imgList.size
        var info : Float = 0f
        var tik : Int
        var tok : Int
        var pre: Duration
        var post : Duration
        if(checkResultDirectories()){
            Log.i(TAG,"Starting benchmark...")
            model.clearResultList()
            if (imgList != null) {
                for ((i, filename) in imgList.take(datasetChunk).withIndex()) {
                    pre = measureTime {
                        model.preprocess(filename)
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
                    model.inferenceOutputToExportFormat()
                    //energyConsumption[i] = tok-tik
                    preTime.add(pre)
                    //activityMainBinding.preTimeVal.text=pre.toString()
                    runTime.add(interpreter.getInferenceTimeNanoseconds())
                    //activityMainBinding.runTimeVal.text=run.toString()
                    postTime.add(post)
                    //activityMainBinding.postTimeVal.text=post.toString()
                    info = (i.toFloat() / datasetChunk * 1e2f)
                    val progressText = getString(R.string.progress,info)
                    withContext(Dispatchers.Main) {
                        activityMainBinding.progress.text = progressText
                        //activityMainBinding.energyVal.text= "${energyConsumption[i]} mAh"
                    }
                }
            }
        }
        writeOutputToFile("${modelPath.split("/").last().split(".").first()}_${interpreter.executingDevice}${model.exportFileExtension}",model.serializeResults())
        val preTimeVal = preTime.reduce { acc, duration -> acc + duration }/datasetChunk
        val runTimeVal = runTime.reduce { acc, duration -> acc + duration }/(datasetChunk*1e6)
        val postTimeVal = postTime.reduce { acc, duration -> acc + duration }/datasetChunk
        //val energyVal = (energyConsumption.reduce { acc, l -> acc + l }).toFloat()/datasetChunk
        val time = "$preTimeVal;$runTimeVal;$postTimeVal"
        writeTimeToFile("${modelPath.split("/").last().split(".").first()}_${interpreter.executingDevice}${model.getTimeFileExtension()}",time)
        withContext(Dispatchers.Main) {
            activityMainBinding.progress.text = getString(R.string.completed)
            //activityMainBinding.preTimeVal.text="AVG: ${preTimeVal}"
            //activityMainBinding.runTimeVal.text="AVG: ${runTimeVal}"
            //activityMainBinding.postTimeVal.text="AVG: ${postTimeVal}"
            //activityMainBinding.energyVal.text="AVG: ${energyVal} uAh"
            activityMainBinding.button.isEnabled = true
        }
    }
    public override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
    private fun writeOutputToFile(filename :String, content: String){
        val dirName = "Results/InferenceResult"
        val dir = File(documentsDir,dirName)
        val file  = File(dir,filename)
        try {
            if(file.exists()){
                file.delete()
            }
            file.createNewFile()
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
            Log.e("FileWrite", "Writing output to file failed $e")
            throw(e)
        }
        Log.i("FileWrite","Output written to file successfully.")
    }
    private fun writeTimeToFile(filename: String,content: String){
        val dirName = "Results/TimeBenchmark"
        val dir = File(documentsDir,dirName)
        val file = File(dir,filename)
        try {
            if(file.exists()){
                file.delete()
            }
            Log.i("FileWrite","Writing time to file...")
            file.writeText(content)
            if(file.readText()!=content){
                throw Exception("Error writing time to file, text mismatch")
            }
        }
        catch (e:Exception) {
            Log.e("FileWrite", "Writing time to file failed $e")
            throw(e)
        }
        Log.i("FileWrite","Time written to file successfully.")
    }
    private fun checkResultDirectories():Boolean{
        var ret = true
        val resultsDir = File(documentsDir,"Results")
        val dirList = listOf("InferenceResult","TimeBenchmark")
        Log.i(TAG,"Checking result directory status...")
        if(!resultsDir.exists()){
            Log.i(TAG,"Results directory missing, creating...")
            if(resultsDir.mkdirs()) {
                Log.i(TAG, "Results directory created successfully, creating subdirectories...")
                for(dir in dirList){
                    val newDir = File(resultsDir,dir)
                    if(!newDir.exists()){
                        Log.i(TAG,"Subdirectory $dir missing. Creating...")
                        if(newDir.mkdirs()){
                            Log.i(TAG,"Directory $dir created successfully.")
                        }
                        else{
                            Log.e(TAG,"Error creating directory $dir.")
                            ret =  false
                        }
                    }
                    else{
                        Log.i(TAG,"Directory $dir already exists.")
                    }
                }
            }
            else{
                Log.e(TAG,"Error creating results directory.")
                ret = false
            }
        }
        else{
            Log.i(TAG,"Results directory already exists. Checking subdirectories...")
            for(dir in dirList){
                val newDir = File(resultsDir,dir)
                if(!newDir.exists()){
                    Log.i(TAG,"Subdirectory $dir missing. Creating...")
                    if(newDir.mkdirs()){
                        Log.i(TAG,"Directory $dir created successfully.")
                    }
                    else{
                        Log.e(TAG,"Error creating directory $dir.")
                        ret = false
                    }
                }
                else{
                    Log.i(TAG,"Directory $dir already exists.")
                }
            }
        }
     return ret
    }
}
