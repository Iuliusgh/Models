package com.example.models

interface ModelInterface {
    fun <T>preprocess(arg:T){}
    fun postprocess(){}
    fun inferenceOutputToExportFormat(){}
    fun serializeResults():String{ return ""}
    fun clearResultList(){}
}
