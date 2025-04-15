package com.example.models

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor.QuantizationParams
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

suspend fun buffer2Array(buffer: ByteBuffer, array: FloatArray, bufferDataType: DataType){
    val byteSize = bufferDataType.byteSize()
    parallelArrayOperation(buffer.capacity()/byteSize,{ i ->
        val byteArray = ByteArray(byteSize)
        val fourByteArray:ByteArray
        for (j in 0 until byteSize){
            byteArray[j] = buffer[i*byteSize+j]
        }
        if(byteSize<4){
            val fill = 4 - byteSize
            fourByteArray = ByteArray(fill) + byteArray
        }
        else{
            fourByteArray = byteArray
        }
        array[i] = ByteBuffer.wrap(fourByteArray).order(ByteOrder.nativeOrder()).float
    })
    /*val chunkSize = buffer.capacity() / (byteSize * numThreads)
    val jobs = mutableListOf<Job>()
    for (i in 0 until numThreads) {
        val start = chunkSize * i
        val end = chunkSize * (i + 1)
        val job = CoroutineScope(Dispatchers.Default).launch {
            val byteArray = ByteArray(byteSize)
            var fourByteArray:ByteArray
            for (index in start until end) {
                for (j in 0 until byteSize){
                    byteArray[j] = buffer[index*byteSize+j]
                }
                if(byteSize<4){
                    val fill = 4 - byteSize
                    fourByteArray = ByteArray(fill) + byteArray
                }
                else{
                    fourByteArray = byteArray
                }
                array[index] = ByteBuffer.wrap(fourByteArray).order(ByteOrder.nativeOrder()).float
            }
        }
        jobs.add(job)
    }
    for (job in jobs) {
        job.join()
    }
    jobs.clear()*/
}
suspend fun array2Buffer(buffer: ByteBuffer, array : FloatArray, bufferDataType : DataType) {
    val byteSize = bufferDataType.byteSize()
    parallelArrayOperation(array.size/byteSize,{ i ->
        if(bufferDataType==DataType.FLOAT32){
            //buffer.put(,array[i].toRawBits().shr(j*8).toByte())
            buffer.putFloat(i*byteSize,array[i])
        }
        else{
            for (j in 0 until byteSize){
                buffer.put(i*byteSize+j,array[i].roundToInt().shr(j*8).toByte())
            }
        }
    })
    /*
    val jobs = mutableListOf<Job>()
    val chunkSize = array.size / numThreads
    for (thread in 0 until numThreads) {
        val start = chunkSize * thread
        val end = chunkSize * (thread + 1)
        val job = CoroutineScope(Dispatchers.Default).launch {
            for (i in start until end){
                if(bufferDataType==DataType.FLOAT32){
                    //buffer.put(,array[i].toRawBits().shr(j*8).toByte())
                    buffer.putFloat(i*byteSize,array[i])
                }
                else{
                    for (j in 0 until byteSize){
                        buffer.put(i*byteSize+j,array[i].roundToInt().shr(j*8).toByte())
                    }
                }
            }
        }
        jobs.add(job)
    }
    for (job in jobs) {
        job.join()
    }
    jobs.clear()*/
    buffer.rewind()
}
suspend fun quantize(array: FloatArray,quant:QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
    /*val jobs = mutableListOf<Job>()
    val chunkSize = array.size / numThreads
    for (thread in 0 until numThreads){
        val start = chunkSize * thread
        val end = chunkSize * (thread + 1)
        val job = CoroutineScope(Dispatchers.Default).launch {
            for(i in start until end){
                array[i] = array[i] / scale + zeroPoint
            }
        }
        jobs.add(job)
    }
    for (job in jobs) {
        job.join()
    }
    jobs.clear()*/
}
suspend fun dequantize(array: FloatArray, quant: QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
    /*
    val jobs = mutableListOf<Job>()
    val chunkSize = buffer.size / numThreads
    for (thread in 0 until numThreads){
        val start = chunkSize * thread
        val end = chunkSize * (thread + 1)
        val job = CoroutineScope(Dispatchers.Default).launch {
            for(i in start until end){
                buffer[i] = (buffer[i] - zeroPoint) * scale
            }
        }
        jobs.add(job)
    }
    for (job in jobs) {
        job.join()
    }
    jobs.clear()*/
}
suspend fun parallelArrayOperation(size:Int,block: (Int) -> Unit,threads:Int=Runtime.getRuntime().availableProcessors()){
    val chunkSize = size / threads
    coroutineScope {
        (0 until threads).map { threadId ->
            async(Dispatchers.Default){
                val start = chunkSize * threadId
                val end = min(start + chunkSize,size)
                for (index in start until end) {
                    block(index)
                }
            }
        }.awaitAll()
    }
}