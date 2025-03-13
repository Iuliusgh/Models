pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        mavenCentral()
        flatDir {
            dirs("src/main/jniLibs")
        }
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "Models"
include(":app")
include(":opencv")

