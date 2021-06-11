import scalax.collection.mutable.Graph
import scalax.collection.GraphPredef._, scalax.collection.GraphEdge._
import scalax.collection.edge.WLkDiEdge
import scalax.collection.edge.Implicits._
import scala.collection.mutable.{Map,HashMap}
import scala.collection._
import scala.math.log
import scalax.collection.GraphTraversal._
import scala.annotation.tailrec
//import fr.acinq.eclair.router.Graph


object Main extends App {

  type Node = String

  // val E=Graph[Node,WLkDiEdge]()

  // E.add(WLkDiEdge("S","A")(2,0))
  // E.add(WLkDiEdge("A","S")(2,0))
  // E.add(WLkDiEdge("S","X")(1,0))
  // E.add(WLkDiEdge("X","S")(1,0))
  // E.add(WLkDiEdge("A","B")(2,0))
  // E.add(WLkDiEdge("B","A")(2,0))
  // E.add(WLkDiEdge("X","B")(9,0))
  // E.add(WLkDiEdge("B","X")(9,0))
  // E.add(WLkDiEdge("X","Y")(7,0))
  // E.add(WLkDiEdge("Y","X")(7,0))
  // E.add(WLkDiEdge("Y","D")(4,0))
  // E.add(WLkDiEdge("D","Y")(4,0))
  // E.add(WLkDiEdge("B","D")(4,0))
  // E.add(WLkDiEdge("D","B")(4,0))

  // val x = time(capacityScalingMinCostFlow("S","D",2,E))
  // val xedges= x.collect{case (e,f) if f !=0 => (e,f,math.exp(-cost(f,e.weight.toLong)))}
  // println(xedges)
  // println("total probability: "+xedges.map(_._3).product)

  case class State(R:Graph[Node,WLkDiEdge],x:Map[WLkDiEdge[Node],Long],e:Map[Node,Long])

  val channelGraph = importChannelGraph()
  val SRC = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
  val DEST = "022c699df736064b51a33017abfc4d577d133f7124ac117d3d9f9633b6297a3b6a"
  val FLOW = 920


  val y=time(capacityScalingMinCostFlow(SRC,DEST,FLOW,channelGraph))
  val edges= y.collect{case (e,f) if f !=0  => (e,f,math.exp(-cost(f,e.weight.toLong).toDouble))}
  println(edges)
  println("total probability: "+edges.map(_._3).product)
  println("with local knowledge at source and destination: "+edges.filter(e => e._1._1!=SRC && e._1._2!=DEST).map(_._3).product)

  @tailrec def scale (G:Graph[Node,WLkDiEdge],delta: Long,state:State,pi:Map[Node,Double]):Map[WLkDiEdge[Node],Long] =
  {
    println(delta+"-scaling phase")
    val R=
      G.edges.toOuter.foldLeft(Graph[Node,WLkDiEdge]())(calculateResidualEdges(_,_,state.x,delta,cost))

    val negEdges = R.edges.toOuter.filter(reducedCost(pi)(_)<0)

    val satState=negEdges.foldLeft(State(R,state.x,state.e))(saturate(G,delta)(_)(_))

    assert(satState.R.edges.toOuter.forall(reducedCost(pi)(_)>=0))

    val (augState,newPi) = augment(satState,pi,delta,G)

    augState.R.edges.toOuter.foreach(e=>assert(reducedCost(newPi)(e)>=0,"edge: "+e+" redcost:"+reducedCost(newPi)(e)+ " pi(x):"+newPi(e._1)+" pi(y):"+newPi(e._2)))

    if (delta == 1) augState.x
    else scale(G,delta/2,augState,newPi)
  }

  @tailrec def augment(state:State,pi:Map[Node,Double],delta:Long,G:Graph[Node,WLkDiEdge]):(State,Map[Node,Double]) =
  {
    val S = state.e.collect {case (n,k) if (k>= delta) => n}
    val T = state.e.collect {case (n,k) if (k<= -delta) => n}
    if (S.isEmpty || T.isEmpty) return (state,pi)

    dijkstraShortestPath(state.R,S.head,T,reducedCost(pi)) match
    {case Some((t,path,distances)) =>
      print("augmenting "+path)
      val newPi=pi.map{case (n:Node,pi:Double) => (n,pi - distances(n))} // everything is reachable
      val augState=path.foldLeft(state)(saturate(G,delta)(_)(_))
      augment(augState,newPi,delta,G)
     case None => throw (new(Exception))//(state,pi)))
    }
  }

  def capacityScalingMinCostFlow(s:Node,d:Node,U:Long,G:Graph[Node,WLkDiEdge]):Map[WLkDiEdge[Node],Long]=
  {
    G.add("null")
    for(node <- G.nodes.toOuter)
    {
      G.add(WLkDiEdge("null",node)(Double.PositiveInfinity,"x"))
      G.add(WLkDiEdge(node,"null")(Double.PositiveInfinity,"y"))
    } //make every node always reachable from every other

    val delta = math.pow(2,(log(U)/log(2)).floor).toLong

    val state=State(Graph[Node,WLkDiEdge](),Map.from(G.edges map {e => (e.toOuter,0L)}),Map.from(G.nodes map {(_,0L)}))
    state.e(s)=U
    state.e(d)= -U

    val pi=Map.from(G.nodes map {n => (n.toOuter,0d)})

    scale(G,delta,state,pi)

    // var total_cnt=0

    // while (delta >= 1)
    // {
    //   val pi = Map.from(G.nodes map {n => (n.toOuter,0d)})
    //   var cnt=0
    //   println(delta+"-scaling phase")

    //   val R=
    //     G.edges.toOuter.foldLeft(Graph[Node,WLkDiEdge]())(calculateResidualEdges(_,_,satState.x,delta,cost))

    //   val negEdges = R.edges.toOuter.filter(reducedCost(pi)(_)<0)

    //   satState=negEdges.foldLeft(State(R,satState.x,satState.e))(saturate(delta)(_)(_))

    //   var S = satState.e.collect {case (n,k) if (k>= delta) => n}.toSet
    //   def T = satState.e.collect {case (n,k) if (k<= -delta) => n}

    //   while (S.size>0 && T.size>0)
    //   {
    //     val s=S.head

    //     val spOpt = dijkstraShortestPath(satState.R,s,T,reducedCost(pi))

    //     for ((t,path,distances) <- spOpt)
    //     {
    //       for {(n,d) <- distances}
    //         pi(n)-=d

    //       satState=path.foldLeft(satState)(saturate(delta)(_)(_))

    //       cnt+=1
    //     }

    //     if (satState.e(s) < delta || spOpt.isEmpty) S-=s //  ensures progress even if no shortest path is found
    //   }

    //   total_cnt+=cnt
    //   println("augmented %d paths in the %d-scaling phase and %d paths in total".format(cnt,delta,total_cnt))

    //   delta/=2

    // }

    // satState.x
 }

  def reducedCost(pi: Map[Node,Double])(edge: WLkDiEdge[Node]):Double =  edge.weight - pi(edge._1) + pi(edge._2)

  def saturate(G:Graph[Node,WLkDiEdge],delta:Long)(s:State)(edge: WLkDiEdge[Node]):State =
  {
      val origEdge = edge.label match {
      case (l:String,true) => G.get(WLkDiEdge(edge._1,edge._2)(0,l)).toOuter
      case (l:String,false) => G.get(WLkDiEdge(edge._2,edge._1)(0,l)).toOuter
    }
    s.x(origEdge) = edge.label match {
      case (l:String,true) => s.x(origEdge)+delta
      case (l:String,false) => s.x(origEdge)-delta
    }
    val R=calculateResidualEdges(s.R,origEdge,s.x,delta,cost)

    s.e(edge._1)-=delta
    s.e(edge._2)+=delta

    State(R,s.x,s.e)

  }



  def cost(a:Long,c:Long): Double = log(c+1)-log(c+1-a)
   // """returns the negative log probability for the success to deliver `a` satoshis through a channel of capacity `c`"""


  def calculateResidualEdges(R:Graph[Node,WLkDiEdge],edge:WLkDiEdge[Node],x:Map[WLkDiEdge[Node],Long],delta:Long,C:(Long,Long)=>Double): Graph[Node,WLkDiEdge] =
  {
    val f = x(edge)
    val cap = edge.weight.round
    val label = edge.label
    val unitCostF=if(cap<Long.MaxValue) (C(f + delta, cap) - C(f,cap))/delta else 100
    val forwardEdge=WLkDiEdge(edge._1,edge._2)(unitCostF,(label,true))
//    assert((!forwardEdge.weight.isNaN()) || f+delta > cap,forwardEdge)
    val unitCostB=if(cap<Long.MaxValue) (C(f - delta, cap) - C(f,cap))/delta else -100
    val backwardEdge=WLkDiEdge(edge._2,edge._1)(unitCostB,(label,false))
    val string=("original: "+edge.toString+" backward: "+backwardEdge.toString+" cap: "+cap+" delta:"+ delta + " f:"+f)
//    assert((!backwardEdge.weight.isNaN())|| delta > f,string)

    if (f+delta <=cap)
      R.upsert(forwardEdge)
    else if (R.contains(forwardEdge)) R.remove(forwardEdge)

  //  assert(!(forwardEdge.weight.isNaN() && R.contains(forwardEdge)),edge+R.get(forwardEdge).toOuter)

    if (delta <= f)
      R.upsert(backwardEdge)
    else if (R.contains(backwardEdge)) R.remove(backwardEdge)

 //   assert(!forwardEdge.weight.isNaN() || !R.contains(forwardEdge))

 //   assert(!backwardEdge.weight.isNaN() || !R.contains(backwardEdge))

    R
  }

  def importChannelGraph():Graph[Node,WLkDiEdge]=
  {
    val jsonString = os.read(os.pwd/"listchannels.json")
    val data = ujson.read(jsonString)

    val G = data("channels").arr.foldLeft(Graph[Node,WLkDiEdge]()){
      (G,channel) =>
      val src = channel("source").str
      val dest = channel("destination").str
      val cap = (channel("satoshis").num) /10000
      val sid = channel("short_channel_id").str
      //    G.add(src)
      //    G.add(dest)
      //     if (G.anyEdgeSelector(G get src,G get dest).isEmpty)
      if (cap>=1d)
        G+=(WLkDiEdge(src,dest)(cap,sid))
      else G
    }
    println("importing "+G.edges.size+" channels")

    G
  }


  def time[R](block: => R): R =
  {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    println("Elapsed time: " + (t1 - t0) + "ms")
    result
  }

  def dijkstraShortestPath(g: Graph[Node,WLkDiEdge],   // based on Acinq's eclair implementation
                           sourceNode: Node,
                           targetNodes: Traversable[Node],
                           cost: WLkDiEdge[Node] => Double
  ): Option[(Node,Seq[WLkDiEdge[Node]],HashMap[Node, Double])] =
  {

    /**
     * This comparator must be consistent with the "equals" behavior, thus for two weighted nodes with
     * the same weight we distinguish them by their public key.
     * See https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html
     */
    case class WeightedNode(key: Node, weight: Double)
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
    val bestWeights = HashMap.newBuilder[Node, Double](initialCapacity, HashMap.defaultLoadFactor).result()
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
          val neighborWeight = cost(edge)+current.weight
          val previousNeighborWeight = bestWeights.getOrElse(neighbor,Double.MaxValue)
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
