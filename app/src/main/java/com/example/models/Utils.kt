package com.example.models


import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor.QuantizationParams
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.function.IntConsumer
import kotlin.concurrent.thread
import kotlin.math.roundToInt

fun array2Buffer(array : FloatArray,buffer: ByteBuffer, bufferDataType : DataType) {
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
fun buffer2Array(buffer: ByteBuffer, array: FloatArray, bufferDataType: DataType){
    buffer.rewind()
    val byteSize = bufferDataType.byteSize()
    //val byteArray = ByteArray(byteSize){0}
    parallelArrayOperation(buffer.capacity()/byteSize,{ i ->
        val byteArray = ByteArray(Float.SIZE_BYTES){0}
        for (j in 0 until byteSize){
            byteArray[j] = buffer[i*byteSize+j]
        }
        if(byteSize<4){
            for (j in 0 until Float.SIZE_BYTES - byteSize){
                byteArray[j] = 0
            }
        }
        array[i] = ByteBuffer.wrap(byteArray).order(ByteOrder.nativeOrder()).getFloat()
    })
}
fun quantize(array: FloatArray,quant:QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
}
fun dequantize(array: FloatArray, quant: QuantizationParams){
    parallelArrayOperation(array.size,{ i ->
        array[i] = array[i] / quant.scale + quant.zeroPoint
    })
}
fun parallelArrayOperation(size:Int, block:IntConsumer, threads:Int=Runtime.getRuntime().availableProcessors()){
    val chunkSize: Int = size / threads
    val jobs = ArrayList<Thread>(threads)
    for (threadId in 0 until threads){
        val start : Int = chunkSize * threadId
        val end : Int = if (threadId < threads - 1) start + chunkSize else size
        val job = thread(start=false){
            var i = start
            while(i < end){
                block.accept(i)
                i++
            }
        }
        jobs.add(job)
        job.start()
    }
    jobs.forEach { it.join() }
}
