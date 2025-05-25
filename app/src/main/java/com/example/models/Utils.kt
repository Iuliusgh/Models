package com.example.models


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
    parallelArrayOperation(buffer.capacity()/4,{ i ->
        for (j in 0 until byteSize){
            GlobalByteArrayArray.byteArrayArray[ThreadId.localThreadId.get()!!][j] = buffer[i*byteSize+j]
        }
        if(byteSize<4){
            for (j in byteSize until Float.SIZE_BYTES){
                GlobalByteArrayArray.byteArrayArray[ThreadId.localThreadId.get()!!][j] = 0
            }
        }
        GlobalByteBufferArray.byteBufferArray[ThreadId.localThreadId.get()!!].clear()
        GlobalByteBufferArray.byteBufferArray[ThreadId.localThreadId.get()!!].put(GlobalByteArrayArray.byteArrayArray[ThreadId.localThreadId.get()!!])
        GlobalByteBufferArray.byteBufferArray[ThreadId.localThreadId.get()!!].rewind()
        array[i] = GlobalByteBufferArray.byteBufferArray[ThreadId.localThreadId.get()!!].getFloat()

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

    for (threadId in 0 until threads){
        val start : Int = chunkSize * threadId
        val end : Int = if (threadId < threads - 1) start + chunkSize else size
        thread(start = false){}
        ThreadPool.pool.execute{
            ThreadId.localThreadId.set(threadId)
            var i = start
            while(i < end){
                block.accept(i)
                i++
            }
            ThreadPool.latch.countDown()
        }
    }
    ThreadPool.latch.await()

}
object ThreadPool{
    private val maxThreads =  Runtime.getRuntime().availableProcessors()
    val pool:ExecutorService = Executors.newFixedThreadPool(maxThreads)
    val latch = CountDownLatch(maxThreads)
    fun destroy(){
        pool.shutdown()
        pool.awaitTermination(1,TimeUnit.MINUTES)
    }
}
object GlobalByteBufferArray{
    val byteBufferArray = Array(Runtime.getRuntime().availableProcessors()){ByteBuffer.allocate(Float.SIZE_BYTES).order(ByteOrder.nativeOrder())}
}
object GlobalByteArrayArray{
    val byteArrayArray = Array(Runtime.getRuntime().availableProcessors()){ ByteArray(Float.SIZE_BYTES){0} }
}
object ThreadId{
    val localThreadId = ThreadLocal<Int>()
}