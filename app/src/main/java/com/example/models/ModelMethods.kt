package com.example.models

interface ModelInterface<T> {
    fun preprocess(arg:T){}
    fun postprocess(){}
}
