package com.example.models

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import com.example.models.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.tensorflow.lite.DataType
import java.io.File


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private val TAG = "Main"
    private lateinit var activityMainBinding: ActivityMainBinding
    private lateinit var model : Model
    private val interpreter = Interpreter(this)
    private var isModelSelected = false
    private var isDeviceSelected = false
    private val placeholder = listOf("---")
    private lateinit var dataset:List<String>
    private val preTime: Array<Long> by lazy { Array(dataset.size){-1} }
    private val runTime:Array<Long> by lazy{ Array(dataset.size){-1L} }
    private val postTime:Array<Long> by lazy{ Array(dataset.size){-1L} }
    private val datasetChunk: Int by lazy { dataset.size }
    //private val energyConsumption = Array(5000) { 0 }
    private val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
    private var info = 0f
    private var nanoTik: Long = 0L
    private var nanoTok: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        if (checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_IMAGES), 0)
        }
        initUI()
        checkResultDirectories()
        //loadDataset()
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

    private suspend fun loop(){
        val modelList = assets.list("models")!!.toList()
        val deviceList = interpreter.getDeviceList()
        for( i in modelList.indices){
            model = when(modelList[i]){
                "YOLO" -> YOLO(this)
                "ResNet" -> ResNet(this)
                else -> break
            }
            val modelVariantList = assets.list(model.modelRootDir)!!.toList().sortedBy{name -> Regex("\\d+").find(name)!!.value.toInt()}
            loadDataset()
            for (j in modelVariantList.indices){
                model.setModelVersion(modelVariantList[j])
                val modelQuantList = assets.list(model.modelRootDir + model.getModelVersion())!!.toList()
                for(k in modelQuantList.indices){
                    model.setModelFullPath(modelQuantList[k].toString())
                    model.setModelName(modelQuantList[k].toString().removeSuffix(".tflite"))
                    model.loadModelFile()
                    for(l in deviceList.indices){
                        interpreter.selectExecutionDevice(l)
                        try{
                            interpreter.initializeOptions()
                        }
                        catch (e:Exception){
                            Log.e(TAG,"Invalid interpreter options, skipping iteration")
                            continue
                        }
                        try {
                            if (interpreter.isInitialized()) {
                                interpreter.close()
                            }
                            interpreter.initializeInterpreter(model)
                            //if(interpreter.getInputDatatype()== DataType.INT16 && l > 1){
                              //  break
                            //}
                            model.initializeIO()
                            validate()
                        } catch (e: Exception) {
                            Log.e("Crash","$e\nSkipping iteration for ${model.getModelName()}_${interpreter.getExecutingDevice()}")
                            continue
                        }
                    }
                    System.gc()
                }
            }
        }
    }
    private fun initUI(){
        //Button to start the validation
        activityMainBinding.button.setOnClickListener {
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    if (interpreter.isInitialized()) {
                        interpreter.close()
                    }
                    interpreter.initializeInterpreter(model)
                }
                catch (e:Exception){
                    Log.e(TAG,"$e")
                }
                model.initializeIO()
                validate()
            }
            activityMainBinding.button.isEnabled = false
            activityMainBinding.progress.text = getString(R.string.starting)
        }
        activityMainBinding.loopButton.setOnClickListener{
            CoroutineScope(Dispatchers.Default).launch {
                loop()
            }
            activityMainBinding.loopButton.isEnabled=false
        }
        //Fill the spinner with the available interpreters
        activityMainBinding.modelSelector.adapter = ArrayAdapter(this,android.R.layout.simple_spinner_item,placeholder + assets.list("models")!!.toList())
        activityMainBinding.device.adapter=ArrayAdapter(this,android.R.layout.simple_spinner_item,placeholder + interpreter.getDeviceList())
        //Attach listeners
        activityMainBinding.modelSelector.onItemSelectedListener = this
        activityMainBinding.modelVersionSelector.onItemSelectedListener = this
        activityMainBinding.modelQuantSelector.onItemSelectedListener = this
        activityMainBinding.device.onItemSelectedListener = this
    }
    override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
        if (parent.getItemAtPosition(position) != placeholder[0]) {
            when (parent.id) {
                activityMainBinding.modelSelector.id -> {
                    activityMainBinding.selectedModel.text = parent.getItemAtPosition(position).toString()
                    try {
                        model = when(parent.getItemAtPosition(position).toString()){
                            "YOLO" -> YOLO(this)
                            "ResNet" -> ResNet(this)
                            else -> throw Exception()
                        }
                        activityMainBinding.modelVersionSelector.adapter = ArrayAdapter(this,android.R.layout.simple_spinner_item, placeholder + assets.list(model.modelRootDir)!!.toList().sortedBy{name -> Regex("\\d+").find(name)!!.value.toInt()})
                        isModelSelected=true
                        loadDataset()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error loading model\n$e")
                    }
                }
                activityMainBinding.modelVersionSelector.id ->{
                    model.setModelVersion(parent.getItemAtPosition(position).toString())
                    activityMainBinding.modelQuantSelector.adapter = ArrayAdapter(this,android.R.layout.simple_spinner_item, placeholder + assets.list(model.modelRootDir + model.getModelVersion())!!.toList())
                }
                activityMainBinding.modelQuantSelector.id -> {
                    model.setModelFullPath(parent.getItemAtPosition(position).toString())
                    model.setModelName(parent.getItemAtPosition(position).toString().removeSuffix(".tflite"))
                    model.loadModelFile()
                }
                activityMainBinding.device.id -> {
                    interpreter.selectExecutionDevice(position - 1)
                    try{
                        interpreter.initializeOptions()
                        //activityMainBinding.datatypeText.text = dataType.toString()
                        isDeviceSelected=true
                    }
                    catch(e:Exception){
                        Log.e(TAG,"$e")
                    }
                }
            }
        }
        if(isModelSelected && isDeviceSelected){
            activityMainBinding.button.isEnabled=true
        }
    }
    override fun onNothingSelected(parent: AdapterView<*>?) {}

    private suspend fun validate() {
        //val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        //var tik : Int
        //var tok : Int

        Log.i(TAG,"Executing ${model.getModelName()} on ${interpreter.getExecutingDevice()}. Starting benchmark...")
        model.clearResultList()
        for (i in 0 until datasetChunk) {
            nanoTik = System.nanoTime()
            model.preprocess(dataset[i])
            if(interpreter.isInputQuantized()){
                quantize(model.modelInput,interpreter.getInputQuant())
            }
            array2Buffer(model.modelInput,interpreter.getInputBuffer(),interpreter.getInputDatatype())
            nanoTok = System.nanoTime()
            preTime[i] = nanoTok-nanoTik
            //tik = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            interpreter.run()
            //tok = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            nanoTik=System.nanoTime()
            buffer2Array(interpreter.getOutputBuffer(),model.modelOutput,interpreter.getOutputDatatype())
            if(interpreter.isOutputQuantized()){
                dequantize(model.modelOutput,interpreter.getOutputQuant())
            }
            model.postprocess()
            nanoTok = System.nanoTime()
            postTime[i] = nanoTok-nanoTik
            interpreter.clearIOBuffers()
            model.inferenceOutputToExportFormat()
            //energyConsumption[i] = tok-tik
            //activityMainBinding.preTimeVal.text=pre.toString()
            runTime[i] = interpreter.getInferenceTimeNanoseconds()
            //activityMainBinding.runTimeVal.text=run.toString()
            //activityMainBinding.postTimeVal.text=post.toString()
            info = (i.toFloat() / datasetChunk * 1e2f)
            withContext(Dispatchers.Main) {
                activityMainBinding.progress.text = getString(R.string.progress,info)
                //activityMainBinding.energyVal.text= "${energyConsumption[i]} mAh"
            }
        }
        writeToFile(outputFilename(),model.serializeResults())
        val preTimeVal = preTime.reduce { acc, duration -> acc + duration }/(datasetChunk*1e6)
        val runTimeVal = runTime.reduce { acc, duration -> acc + duration }/(datasetChunk*1e6)
        val postTimeVal = postTime.reduce { acc, duration -> acc + duration }/(datasetChunk*1e6)
        //val energyVal = (energyConsumption.reduce { acc, l -> acc + l }).toFloat()/datasetChunk
        val time = "${preTimeVal}ms;${runTimeVal}ms;${postTimeVal}ms"
        writeToFile(outputFilename(true),time,true)
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
        ThreadPool.destroy()
        super.onDestroy()
    }
    private fun outputFilename(isTime:Boolean = false):String{
        val filename = if(!isTime){ "${model.getModelName()}_${interpreter.getExecutingDevice()}_output${model.exportFileExtension}" } else{ "${model.getModelName()}_${interpreter.getExecutingDevice()}_time${model.getTimeFileExtension()}" }
        return filename
    }
    private fun writeToFile(filename :String, content: String, isTime: Boolean = false){
        val dirName = if(!isTime){ "Results/InferenceResult" } else{ "Results/TimeBenchmark" }
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
        }
        catch (e:Exception) {
            Log.e("FileWrite", "Writing output to file failed $e")
            throw(e)
        }
        Log.i("FileWrite","Output written to $filename successfully.")
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
    private fun listDataset(){
        Log.i(TAG,"Listing validation dataset...")
        //val imgList = File(model.datasetPath).walkTopDown().filter{it.extension == "jpg"}.map { it.absolutePath }.toList().sorted()
        Log.i(TAG,"Finished listing.")
        val f = File(documentsDir,"a.txt")
        //f.writeText(imgList.joinToString("\n"))
    }
    private fun loadDataset(){
        Log.i(TAG,"Loading dataset...")
        dataset = assets.open(model.datasetPaths).bufferedReader().readLines()//.take(5)
        Log.i(TAG,"Dataset paths loaded.")
    }
}
