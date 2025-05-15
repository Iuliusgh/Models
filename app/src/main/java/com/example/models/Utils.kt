package com.example.models


import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor.QuantizationParams
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
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
suspend fun parallelArrayOperation(size:Int, block: (Int) -> Unit, threads:Int=Runtime.getRuntime().availableProcessors()){
    val chunkSize = size / threads
    coroutineScope {
        (0 until threads).map { threadId ->
            val start = chunkSize * threadId
            val end = min(start + chunkSize,size)
            async(Dispatchers.Default){
                for (index in start until end) {
                    block(index)
                }
            }
        }.awaitAll()
    }
}
