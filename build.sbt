name := "SparkMediumAnalyse"

version := "1.0"

scalaVersion := "2.11.8"

lazy val sparkVersion = "2.1.0"

resolvers ++= Seq(
  "conjars" at "http://conjars.org/repo",
  "clojars" at "http://clojars.org/repo",
  "jitpack" at "https://jitpack.io"
)
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" % "spark-mllib_2.11" % sparkVersion % "provided",
  //  "org.apache.spark" % "spark-mllib_2.11" % sparkVersion,
  "com.typesafe" % "config" % "1.3.1",
  "org.apache.hadoop" % "hadoop-aws" % "2.7.0",
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.2.3"
)

dependencyOverrides += "com.databricks" % "spark-avro_2.11" % "3.2.0"
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _ *) => MergeStrategy.discard
  case x => MergeStrategy.first
}
