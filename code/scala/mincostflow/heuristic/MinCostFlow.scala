// Copyright 2021 René Pickhardt and Stefan Richter

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// this Scala code is derived from the original python code in
// https://github.com/renepickhardt/mpp-splitter/code/simulation/MinCostMaxFlowScalingAugmentingPaths.ipynb
//
// it contains code derived from Acinq's eclair implementation at
// https://github.com/ACINQ/eclair/blob/ebed5ad9ea27771f48d871e0ff5bf5780bd1925c/eclair-core/src/main/scala/fr/acinq/eclair/router/Graph.scala
// which is under Copyright 2019 ACINQ SAS (see below)

// this version implements the fast heuristic discovered while
// trying to implement the algorithm from Chapter 14.5 in the book
// Network Flows: Theory, Algorithms, and Applications
// by Ahuja, Magnanti and Orlin
//
// it uses Long costs for increased numerical stability


package mincostflow
import scalax.collection.mutable.Graph
import scalax.collection.GraphPredef._, scalax.collection.GraphEdge._
import scalax.collection.edge.WLkDiEdge
import scalax.collection.edge.Implicits._
import scala.collection.mutable.{Map,HashMap}
import scala.collection._
import scala.math.log
import scalax.collection.GraphTraversal._


object MinCostFlow extends App {

  type Node = String

  val (channelGraph, fees) = importChannelGraph()
  val SRC = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
  val DEST = "022c699df736064b51a33017abfc4d577d133f7124ac117d3d9f9633b6297a3b6a"
  val FLOW = 920

  val minBalance = Map.from(channelGraph.edges.toOuter map
                              {e => (e, if (e._1 != SRC) 0L else e.weight.toLong)})
  val inFlight = Map.from(channelGraph.edges.toOuter map (e => (e,0L)))

  val y=time(capacityScalingMinCostFlow(SRC,DEST,FLOW,channelGraph,minBalance,inFlight,fees))
  val edges= y.collect{case (e,f) if f !=0 =>
    val effAmt = Math.max(0,inFlight(e)+f-minBalance(e))
    val effCap = channelGraph.get(e).weight.toLong+1-minBalance(e)
    val prob:Double=(effCap-effAmt).toDouble/effCap
    (e,f,prob,fees(e)*f)}
  for (e <- edges)
    println(List(e._1.label,e._1._1,e._1._2,e._2,e._3,e._4,inFlight(e._1),minBalance(e._1),e._1.weight).mkString(","))
  //MFCround-rdNr: flow edges including sid, src, dest, flow, prob, fees, inFlight, minBalance, capacity
  println("total probability: "+edges.map(_._3).iterator.product)


  println("total fees: "+edges.filter(e => e._1._1!=SRC).map(_._4).sum)



  def capacityScalingMinCostFlow(s:Node,d:Node,U:Long,G:Graph[Node,WLkDiEdge],minBalance:Map[WLkDiEdge[Node],Long], inFlight: Map[WLkDiEdge[Node],Long],fees: Map[WLkDiEdge[Node],Double]):Map[WLkDiEdge[Node],Long]=
  {
    var x: Map[WLkDiEdge[Node],Long]=Map.from(G.edges map {e => (e.toOuter,0L)})
    var e: Map[Node,Long]=Map.from(G.nodes map {(_,0L)})

    e(s)=U
    e(d)= -U

    var delta = math.pow(2,(log(U)/log(2)).floor).toLong


    var total_cnt=0


    while (delta >= 1)
    {
      var cnt=0
      println(delta+"-scaling phase")

      val pi = Map.from(G.nodes map {n => (n.toOuter,0L)})

//      val correctors = Map.from(G.edges map {e => (e.toOuter,0d)}) // for kahan Addition

      val tup = saturatedDeltaResidualNetwork(G,minBalance,inFlight,x,delta,cost,e,fees,s)
      val R = tup._1
      val costs = tup._2

      var S = e.collect {case (n,k) if (k>= delta) => n}
      def T = e.collect {case (n,k) if (k<= -delta) => n}

      while (S.size>0 && T.size>0)
      {
        val s=S.head
        val spOpt = dijkstraShortestPath(R,s,T,reducedCost)

        for ((t,path,distances) <- spOpt)
        {
          augment(path)

          for (n <- G.nodes)
            pi(n)-=distances.getOrElse(n,10000000000L)

          e(s)-=delta
          e(t)+=delta
          cnt+=1
        }

        if (e(s) < delta || spOpt.isEmpty) S=S.tail //  ensures progress if no shortest path is found
      }

      total_cnt+=cnt
      println("augmented %d paths in the %d-scaling phase and %d paths in total".format(cnt,delta,total_cnt))

      delta/=2



      def reducedCost(edge: WLkDiEdge[Node]):Long = {
        val x = costs(edge) - (pi(edge._1)) + (pi(edge._2))

        assert(x>=0,"c:"+edge.weight+" -pi(e1):"+(-delta*pi(edge._1))+" pi(edge._2):"+delta*pi(edge._2)+" x:"+x+edge)
        x
      }

      def augment(path:Seq[WLkDiEdge[Node]]) =
      { //beware: changes R,x
        for {edge <- path}
        {
          val olabel = edge.label match { case (l,i) => l }
          val originalEdge = WLkDiEdge(edge._1,edge._2)(0,olabel)
          val originalCounterEdge = WLkDiEdge(edge._2,edge._1)(0,olabel)
          if (x.contains(originalCounterEdge) && x(originalCounterEdge)>=delta)
            x(originalCounterEdge)-=delta
          else x(originalEdge)=x.getOrElse(originalEdge,0L)+delta

          val counterEdge=WLkDiEdge(edge._2,edge._1)(-edge.weight,edge.label)
          R add counterEdge
          costs(counterEdge)= -costs(edge)
          R remove edge
        }

      }

    }

    x
 }

  def mu = 0.0001
  def cost(minBalance:Long,inFlight:Long,c:Long,fee: Double,a:Long): Long ={
    assert(a+inFlight<=c, "inFlight: "+inFlight+"a: "+a+"c: "+c)
    val effAmt = Math.max(0,inFlight+a-minBalance)
    val effCap = c+1-minBalance
    (1000000d*(log(effCap.toDouble/(effCap-effAmt).toDouble)+mu*a*fee)).toLong // no loss of significance?
    //log(effCap)-log(effCap-effAmt)+mu*a*fee
  }
    // fee in satoshis
   // """returns the negative log probability for the success to deliver `a` satoshis through a channel of capacity `c`"""



  def saturatedDeltaResidualNetwork
    (G:Graph[Node,WLkDiEdge], minBalance:Map[WLkDiEdge[Node],Long], inFlight: Map[WLkDiEdge[Node],Long] ,x:Map[WLkDiEdge[Node],Long],delta:Long,cost:(Long,Long,Long,Double,Long)=>Long,e: Map[Node,Long],fees:Map[WLkDiEdge[Node],Double],s:Node):
      (Graph[Node,WLkDiEdge],Map[WLkDiEdge[Node],Long]) =
  {  // beware: changes x and e
    val costs = Map[WLkDiEdge[Node],Long]()

    val R=Graph[Node,WLkDiEdge]()
    for (edge <- G.edges.toOuter)
    {
      val f = x(edge)
      val cap = edge.weight.toLong
      val label =edge.label
      val C=cost(minBalance(edge),inFlight(edge),cap,fees(edge),_)

      if (f+delta <=cap-inFlight(edge))
        {
          val unitCost = (C(f + delta) - C(f))/delta
          R add WLkDiEdge(edge._1,edge._2)(unitCost,(label,0))
          costs(WLkDiEdge(edge._1,edge._2)(unitCost,(label,0)))=unitCost
        }
      if (delta <= f) //backward flow
        {
          val unitCost = (C(f - delta) - C(f))/delta
          R add WLkDiEdge(edge._1,edge._2)(-unitCost,(label,1))
          costs(WLkDiEdge(edge._1,edge._2)(-unitCost,(label,1)))= -unitCost
          x(edge)-=delta // presaturate the counteredge
          e(edge._2)-=delta
          e(edge._1)+=delta
          // this directly inserts the counter edge after saturating the backward flow
          // it's a shortcut that makes an extra saturation step unnecessary
          // under the assumption of non-negative weights.
          // if negative weights are needed, add an explicit saturation step for negative cost edges
        }
    }
    (R,costs)
  }


  def importChannelGraph():(Graph[Node,WLkDiEdge],Map[WLkDiEdge[Node],Double])=
  {
    val jsonString = os.read(os.pwd/"listchannels.json")
    val data = ujson.read(jsonString)

    val G = Graph[Node,WLkDiEdge]()
    val fees = Map[WLkDiEdge[Node],Double]()

    for (channel <- data("channels").arr)
    {
      val src = channel("source").str
      val dest = channel("destination").str
      val cap = (channel("satoshis").num) /10000
      val sid = channel("short_channel_id").str
      val fee = channel("fee_per_millionth").num / 100 // per 10000
      G.add(src)
      G.add(dest)
      //if (G.anyEdgeSelector(G get src,G get dest).isEmpty)
      G.add(WLkDiEdge(src,dest)(cap,sid))
      fees(WLkDiEdge(src,dest)(cap,sid))=fee
    }
    println("importing "+G.edges.size+" channels")

    (G,fees)
  }


  def time[R](block: => R): R =
  {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    println("Elapsed time: " + (t1 - t0) + "ms")
    result
  }

// The code in the following method is derived from Acinq's eclair implementation at
// https://github.com/ACINQ/eclair/blob/ebed5ad9ea27771f48d871e0ff5bf5780bd1925c/eclair-core/src/main/scala/fr/acinq/eclair/router/Graph.scala
// which is under Copyright 2019 ACINQ SAS
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
  def dijkstraShortestPath(g: Graph[Node,WLkDiEdge],   // based on Acinq's eclair implementation
                           sourceNode: Node,
                           targetNodes: Traversable[Node],
                           cost: WLkDiEdge[Node] => Long
  ): Option[(Node,Seq[WLkDiEdge[Node]],HashMap[Node, Long])] =
  {

    /**
     * This comparator must be consistent with the "equals" behavior, thus for two weighted nodes with
     * the same weight we distinguish them by their public key.
     * See https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html
     */
    case class WeightedNode(key: Node, weight: Long)
    object NodeComparator extends Ordering[WeightedNode] {
      override def compare(x: WeightedNode, y: WeightedNode): Int = {
        val weightCmp = x.weight.compareTo(y.weight)
        if (weightCmp == 0) x.key.toString().compareTo(y.key.toString())
        else weightCmp
      }
    }



    val sourceNotInGraph = !g.contains(sourceNode)
    if (sourceNotInGraph) {
      return None
    }

    // conservative estimation to avoid over-allocating memory: this is not the actual optimal size for the maps,
    // because in the worst case scenario we will insert all the vertices.
    val initialCapacity = 1000
    val bestWeights = HashMap.newBuilder[Node, Long](initialCapacity, HashMap.defaultLoadFactor).result()
    val bestEdges = HashMap.newBuilder[Node, WLkDiEdge[Node]](initialCapacity, HashMap.defaultLoadFactor).result()
    // NB: we want the elements with smallest weight first, hence the `reverse`.
    val toExplore = mutable.PriorityQueue.empty[WeightedNode](NodeComparator.reverse)
    val visitedNodes = mutable.HashSet[Node]()

    // initialize the queue and cost array with the initial weight
    bestWeights.put(sourceNode, 0)
    toExplore.enqueue(WeightedNode(sourceNode, 0))

    while (toExplore.nonEmpty) {
      // node with the smallest distance from the target
      val current = toExplore.dequeue() // O(log(n))
      if (!visitedNodes.contains(current.key)) {
        visitedNodes += current.key
        val currentNode=g get current.key
        val neighborEdges = (g get current.key).outgoing.map(_.toOuter)
        neighborEdges.foreach { edge =>
          val neighbor = edge._2
       //   assert(cost(edge)>=0,edge+" has negative cost "+cost(edge))
          val neighborWeight = cost(edge)+current.weight
          val previousNeighborWeight = bestWeights.getOrElse(neighbor,Long.MaxValue)
            // if this path between neighbor and the target has a shorter distance than previously known, we select it
          if (neighborWeight < previousNeighborWeight) {
              // update the best edge for this vertex
              bestEdges.put(neighbor, edge)
              // add this updated node to the list for further exploration
              toExplore.enqueue(WeightedNode(neighbor, neighborWeight)) // O(1)
              // update the minimum known distance map
              bestWeights.put(neighbor, neighborWeight)
          }
        }
      }
    }

    targetNodes.collectFirst
    { case targetNode if (bestEdges.contains(targetNode)) =>
      {
      val edgePath = new mutable.ArrayBuffer[WLkDiEdge[Node]](20)//max-length
      var current = bestEdges.get(targetNode)
      while (current.isDefined) {
        edgePath += current.get
        current = bestEdges.get(current.get._1)
        }
      (targetNode,edgePath.toSeq,bestWeights)
      }
    }

    }
}
