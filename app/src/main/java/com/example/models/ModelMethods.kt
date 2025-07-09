package com.example.models

interface ModelInterface {
    fun <T>preprocess(arg:T){}
    suspend fun postprocess(){}
    fun inferenceOutputToExportFormat(){}
    fun serializeResults():String{ return ""}
    fun clearResultList(){}
}
