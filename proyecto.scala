// Contenido del proyecto
// 1.Objetivo: Comparaciones de rendimiento de los siguientes algoritmos de aprendizaje automático
// - K-medias
// - Bisecar K-medias
// - Con el conjunto de datos iris

//k-medias
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
import org.apache.spark.ml.clustering.KMeans
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","label")

val lis = df.select("label").map(r => r.getString(0)).collect.toList
var l= 0.0 :: Nil
for(label <- lis){
  if(label =="Iris-setosa"){
    l= l ::: List(1.0)
  }
  if(label =="Iris-versicolor"){
    l= l ::: List(2.0)
  }
  if(label =="Iris-virginica"){
    l= l ::: List(3.0)
  }
}

val dato = l.drop(1).toDS().toDF()
val dflabel = dato.withColumn("label",(dato("value"))).select("label")
val dffeatures = df.withColumn("sepallength", (df("sepallength"))).withColumn("sepalwidth", (df("sepalwidth"))).withColumn("petallength", (df("petallength"))).withColumn("petalwidth", (df("petalwidth"))).drop("label")
//val dffeatures = dr.withColumn("features", (df("features"))).select("features")

import org.apache.spark.sql.functions.monotonicallyIncreasingId

val dflabels = dflabel.withColumn("id", monotonicallyIncreasingId)
val dffeaturess = dffeatures.withColumn("idd", monotonicallyIncreasingId)

val dg = dflabels.join(dffeaturess,col("id") === col("idd"),"inner")
val feature_data = dg.drop("id","idd")

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
val Model_VectorAssembler= new VectorAssembler().setInputCols(Array("sepallength","sepalwidth","petallength","petalwidth","label")).setOutputCol("features")
val training = Model_VectorAssembler.transform(feature_data).select("features")
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training)
val WSSSE = model.computeCost(training)
println(s"Resultados de Within Set Sum of Squared Errors: ${WSSSE} ")
println("Cluster Centers: ")
model.clusterCenters.foreach(println)


//// - Bisecar K-medias

import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","class")
val newcol = when($"class".contains("Iris-setosa"), 1.0).otherwise(when($"class".contains("Iris-virginica"), 3.0).otherwise(2.0))
val newdf = df.withColumn("ID", newcol)
newdf.select("ID","SepalLength","SepalWidth","PetalLength","PetalWidth","class").show(150, false)
val assembler = new VectorAssembler().setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth","ID")).setOutputCol("features")
val features = assembler.transform(newdf)
features.show(5)
val bkmeans = new BisectingKMeans().setK(2).setSeed(1)
val model = bkmeans.fit(features)
val cost = model.computeCost(features)
println(s"Within Set Sum of Squared Errors = $cost")
println("Cluster Centers: ")
val centers = model.clusterCenters
centers.foreach(println)
