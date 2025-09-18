package com.example.models

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Model
import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

open class Model(private val context: Context):ModelInterface {
    private val TAG = "Model"
    private var loaded = false
    open val modelRootDir:String = "models/"
    open val datasetPaths:String
        get() = "datasetPaths/"
    open val exportFileExtension:String
        get() = ""
    //private lateinit var modelBuffer:MappedByteBuffer
    lateinit var inputShape : List<Int>
    lateinit var outputShape : List<Int>
    lateinit var modelInput :FloatArray
    lateinit var modelOutput :FloatArray
    private lateinit var modelName:String
    private val timeFileExtension = ".csv"
    private lateinit var modelFullPath:String
    private var modelVersion:String = ""
    private lateinit var loadedModel : Model

    fun loadModelFile() {
        loaded = false
        try{
            loadedModel = Model.load(context.assets,modelFullPath)
            /*val fd = context.assets.openFd(modelFullPath)
            val fileChannel = FileInputStream(fd.fileDescriptor).channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY,fd.startOffset,fd.declaredLength)
            modelBuffer.order(ByteOrder.nativeOrder())*/
            Log.i(TAG,"Loaded model $modelName")
        }
        catch (e: UninitializedPropertyAccessException){
            Log.e(TAG,"Error trying to load model:\n$e")
            throw e
        }
        loaded = true
    }
    fun setLayout(){
        inputShape = loadedModel.getInputTensorType("args_0").layout!!.dimensions
        outputShape = loadedModel.getOutputTensorType("output_0").layout!!.dimensions
    }
/*
    fun getModelBuffer():MappedByteBuffer {
        if (!loaded) {
            throw Exception("No model currently loaded.")
        }
        return modelBuffer
    }*/
    fun getLoadedModel(): Model{
        if (!loaded){
            throw Exception("No model currently loaded.")
        }
        return loadedModel
    }
    /*
    fun setIOShape(inShape:IntArray, outShape:IntArray){
        inputShape = inShape
        outputShape = outShape
    }*/
   fun initializeIO(){
        modelInput = FloatArray(inputShape.reduce{acc,i -> acc * i})
        modelOutput = FloatArray(outputShape.reduce{acc,i -> acc * i})
    }
    fun getTimeFileExtension():String{
        return timeFileExtension
    }
    fun setModelFullPath(path:String){
        modelFullPath= "$modelRootDir$modelVersion/$path"
    }
    fun getModelFullPath():String{
        return modelFullPath
    }
    fun setModelName(name:String){
        modelName = name
    }
    fun getModelName():String{
        return modelName
    }
    fun setModelVersion(version:String){
        modelVersion=version
    }
    fun getModelVersion():String{
        return modelVersion
    }
}
