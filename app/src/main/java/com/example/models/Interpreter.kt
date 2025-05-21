package com.example.models

import android.content.Context
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor.QuantizationParams
import java.lang.Runtime.getRuntime
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Interpreter (private val context: Context){
    private var initialized = false
    private val deviceList = queryDeviceCapabilities()
    private lateinit var executingDevice : String
    /*: Map<Int,String> = mapOf(
        0 to "CPU - Single core",
        1 to "CPU - Multicore",
        2 to "GPU - Float32",
        3 to "GPU - Float16",
        4 to "NPU - Int8",
        5 to "NPU - Float16"
    )*/
    private val interpreterOptions = Interpreter.Options()
    private lateinit var liteRTInterpreter: Interpreter
    data class IOInfo(
        var dataType:DataType,
        var quant:QuantizationParams,
        var shape:IntArray,
    )
    private lateinit var inputInfo:IOInfo
    private lateinit var outputInfo:IOInfo
    private lateinit var inputBuffer:ByteBuffer
    private lateinit var outputBuffer:ByteBuffer

    private fun queryDeviceCapabilities(): MutableList<String> {
        val deviceCapabilities = mutableListOf("CPU_SC")
        if(getRuntime().availableProcessors()>1){
            deviceCapabilities.add("CPU_MC")
        }
        if(QnnDelegate.checkCapability(QnnDelegate.Capability.GPU_RUNTIME)){
            deviceCapabilities.add("GPU_FP32")
            deviceCapabilities.add("GPU_FP16")
        }
        if(QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME)){
            deviceCapabilities.add("DSP")
        }
        if(QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)){
            deviceCapabilities.add("HTP_IQ")
        }
        if(QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)){
            deviceCapabilities.add("HTP_FP16")
        }
        return deviceCapabilities
    }
    fun initializeOptions() {
        interpreterOptions.runtime = InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY
        interpreterOptions.setAllowBufferHandleOutput(true)
        interpreterOptions.setUseNNAPI(false)
        interpreterOptions.setUseXNNPACK(false)
        when (executingDevice) {
            "CPU_SC" -> {
                interpreterOptions.setUseXNNPACK(true)
                interpreterOptions.setNumThreads(1)
            }

            "CPU_MC" -> {
                interpreterOptions.setUseXNNPACK(true)
                interpreterOptions.setNumThreads(getRuntime().availableProcessors())
            }

            else -> {
                try {
                    interpreterOptions.addDelegate(initQNNDelegate())
                    Log.i("Interpreter", "QnnDelegate initialized successfully")
                } catch (e: UnsupportedOperationException) {
                    Log.e("Interpreter", "Error during QnnDelegate initialization\n$e")
                }
            }
        }
    }
    private fun initializeIOInfo(){
        inputInfo = IOInfo(liteRTInterpreter.getInputTensor(0).dataType(),
            liteRTInterpreter.getInputTensor(0).quantizationParams(),
            liteRTInterpreter.getInputTensor(0).shape())
        outputInfo= IOInfo(liteRTInterpreter.getOutputTensor(0).dataType(),
            liteRTInterpreter.getOutputTensor(0).quantizationParams(),
            liteRTInterpreter.getOutputTensor(0).shape())
    }
    private fun initializeIOBuffers(){
        inputBuffer = ByteBuffer.allocateDirect(inputInfo.shape.reduce{acc,i -> acc * i} * inputInfo.dataType.byteSize())
        inputBuffer.order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputInfo.shape.reduce{acc,i -> acc * i} * outputInfo.dataType.byteSize())
        outputBuffer.order(ByteOrder.nativeOrder())
    }
    fun initializeInterpreter(model:Model){
        try {
            liteRTInterpreter = Interpreter(model.getModelBuffer(), interpreterOptions)
            Log.i("Delegate","Delegate instantiated successfully using model")
            initialized=true
        }
        catch (e: Exception){
            Log.e("Interpreter","Cannot initialize TFLiteInterpreter",e)
            throw e
        }
        initializeIOInfo()
        initializeIOBuffers()
        model.setIOShape(inputInfo.shape,outputInfo.shape)
        Log.i("Interpreter","Initialized input and output buffers")
    }
    private fun initQNNDelegate(): Delegate {
        val options = QnnDelegate.Options()
        //options.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_VERBOSE)
        options.skelLibraryDir = context.applicationInfo.nativeLibraryDir
        //options.libraryPath = applicationInfo.nativeLibraryDir
        options.cacheDir = context.cacheDir.absolutePath
        when(executingDevice){
            "GPU_FP32" -> {
                options.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND)
                options.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP32)
                options.setGpuPerformanceMode(QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH)
            }
            "GPU_FP16" -> {
                options.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND)
                options.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16)
                options.setGpuPerformanceMode(QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH)
            }
            "DSP" -> {
                options.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND)
                options.setDspPerformanceMode(QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_BURST)
            }
            "HTP_IQ" -> {
                options.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                options.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
                options.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
                options.setHtpUseFoldRelu(QnnDelegate.Options.HtpUseFoldRelu.HTP_FOLD_RELU_ON)
                options.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            }
            "HTP_FP16" -> {
                options.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                options.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                options.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
                options.setHtpUseFoldRelu(QnnDelegate.Options.HtpUseFoldRelu.HTP_FOLD_RELU_ON)
                options.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            }
        }
        return  QnnDelegate(options)
    }
    fun getDeviceList():List<String>{
        return deviceList
    }
    fun isInitialized():Boolean{
        return initialized
    }
    fun close(){
        liteRTInterpreter.close()
        initialized=false
    }
    fun run(){
        liteRTInterpreter.run(inputBuffer,outputBuffer)
    }
    fun selectExecutionDevice(device:Int){
        executingDevice=deviceList[device]
    }
    fun isInputQuantized():Boolean{
        return inputInfo.quant.scale != 0.0f
    }
    fun getInputQuant():QuantizationParams{
        return inputInfo.quant
    }
    fun isOutputQuantized():Boolean{
        return outputInfo.quant.scale != 0.0f
    }
    fun getOutputQuant():QuantizationParams{
        return outputInfo.quant
    }
    fun getInputDatatype():DataType{
        return inputInfo.dataType
    }
    fun getOutputDatatype():DataType{
        return outputInfo.dataType
    }
    fun getInputBuffer():ByteBuffer{
        return inputBuffer
    }
    fun getOutputBuffer():ByteBuffer{
        return outputBuffer
    }
    fun clearIOBuffers(){
        inputBuffer.clear()
        outputBuffer.clear()
    }
    fun getInferenceTimeNanoseconds():Long{
        return liteRTInterpreter.lastNativeInferenceDurationNanoseconds
    }
    fun getExecutingDevice():String{
        return executingDevice
    }
}


