ThisBuild / organization := "com.example"
ThisBuild / scalaVersion := "2.13.6"
ThisBuild / version      := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "MinCostFlow",
    fork:= true,
    libraryDependencies += "org.scala-graph" %% "graph-core" % "1.13.2",
    libraryDependencies += "com.lihaoyi" %% "upickle" % "1.3.15",
    libraryDependencies += "com.lihaoyi" %% "os-lib" % "0.7.8"
  )
