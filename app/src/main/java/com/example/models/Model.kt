package com.example.models

import android.content.Context
import android.util.Log
import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

open class Model<T>(private val context: Context):ModelInterface<T> {
    private val TAG = "Model"
    private var loaded = false
    private lateinit var modelBuffer:MappedByteBuffer
    protected lateinit var inputShape:IntArray
    private lateinit var outputShape:IntArray
    protected var modelInput = FloatArray(inputShape.reduce{acc,i -> acc * i})
    private var modelOutput = FloatArray(outputShape.reduce{acc,i -> acc * i})

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
    fun getModelInput():FloatArray{
        return modelInput
    }
    fun getModelOutput():FloatArray{
        return modelOutput
    }
}
