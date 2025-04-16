package com.example.models

import java.io.File

interface ModelInterface {
    fun <T>preprocess(arg:T){}
    suspend fun postprocess(){}
    fun inferenceOutputToExportFormat(filename:String){}
    fun serializeResults():String{ return ""}
}
