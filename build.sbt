name := "twitter-sentiment"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.2"

libraryDependencies ++= Seq(
  //...
  "com.github.catalystcode" %% "streaming-rss-html" % "1.0.2",
  //...
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

