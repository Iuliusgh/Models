package com.example.models


import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor.QuantizationParams
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.function.IntConsumer
import kotlin.ByteArray
import kotlin.concurrent.thread
import kotlin.math.roundToInt

suspend fun array2Buffer(array : FloatArray,buffer: ByteBuffer, bufferDataType : DataType) {
    val byteSize = bufferDataType.byteSize()
    parallelArrayOperation(array.size,{ i ->
        if(bufferDataType==DataType.FLOAT32){
            buffer.putFloat(i*byteSize,array[i])
        }
        else{
            for (j in 0 until byteSize){
                buffer.put(i*byteSize+j,array[i].roundToInt().shr(j*8).toByte())
            }
        }
    })
    buffer.rewind()
}
suspend fun buffer2Array(buffer: ByteBuffer, array: FloatArray, bufferDataType: DataType){
    buffer.rewind()
    val byteSize = bufferDataType.byteSize()
    parallelArrayOperation(array.size,{ i ->
        when(byteSize){
            1 -> {
                array[i] = buffer.get(i).toFloat()
            }
            2 -> {
                array[i] = buffer.getShort(i*byteSize).toFloat()
            }
            4 -> {
                array[i] = buffer.getFloat(i*byteSize)
            }
        }
    })
}
suspend fun quantize(array: FloatArray,quant:QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
}
suspend fun dequantize(array: FloatArray, quant: QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
}
suspend inline fun parallelArrayOperation(size: Int, block:IntConsumer, threads: Int = Runtime.getRuntime().availableProcessors()) {
    val chunkSize: Int = size / threads
    coroutineScope {
        for (threadId in 0 until threads) {
            val start: Int = chunkSize * threadId
            val end: Int = if (threadId < threads - 1) start + chunkSize else size
            launch(Dispatchers.Default) {
                for (i in start until end) {
                    block.accept(i)
                }
            }
        }
    }
}