package com.example.models

import android.content.Context
import android.util.Log
import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

open class Model(private val context: Context):ModelInterface {
    private val TAG = "Model"
    private var loaded = false
    open val datasetPath:String = "/storage/emulated/0/Dataset/"
    open val exportFileExtension:String = ""
    private lateinit var modelBuffer:MappedByteBuffer
    protected lateinit var inputShape:IntArray
    protected lateinit var outputShape:IntArray
    lateinit var modelInput :FloatArray
    lateinit var modelOutput :FloatArray
    private lateinit var modelName:String
    private val timeFileExtension = ".csv"

    fun loadModelFile(path:String) {
        loaded = false
        try{
            val fd = context.assets.openFd(path)
            val fileChannel = FileInputStream(fd.fileDescriptor).channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY,fd.startOffset,fd.declaredLength)
            modelBuffer.order(ByteOrder.nativeOrder())
        }
        catch (e:Exception){
            Log.e(TAG,"Error trying to load model:\n$e")
        }
        loaded = true
    }
    fun getModelBuffer():MappedByteBuffer {
        if (!loaded) {
            throw Exception("No model currently loaded.")
        }
        return modelBuffer
    }
    fun setIOShape(inShape:IntArray, outShape:IntArray){
        inputShape = inShape
        outputShape = outShape
    }
    fun initializeIO(){
        modelInput = FloatArray(inputShape.reduce{acc,i -> acc * i})
        modelOutput = FloatArray(outputShape.reduce{acc,i -> acc * i})
    }
    fun getTimeFileExtension():String{
        return timeFileExtension
    }
}
