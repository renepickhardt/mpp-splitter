# Optimally Reliable & Cheap Payment Flows on the Lightning Network

This is joint work work with [Stefan Richter](https://twitter.com/stefanwouldgo). It was initiated as follow-up research of the [probabilistic payment routing paper](https://arxiv.org/abs/2103.08576) on the Lightning Network and addresses the question of how to optimally conduct a multipart payment (MPP) split. Hence the repository name `mpp-splitter`. 

The paper that you find in this repository also exists as a [pdf preprint on arxiv.org](https://arxiv.org/abs/2107.05322).

The 2 main ideas are to quantify the uncertainty of channel balances with the help of probability theory and to transform the problem of finding the most optimal mpp split (with respect to high likelihood and low fees) to an [integer min-cost flow problem with a separable convex cost function](https://twitter.com/renepickhardt/status/1385144337907044352https://twitter.com/renepickhardt/status/1385144337907044352).

This result naturally leads to a round-based algorithm of min-cost flow computations by updating the uncertainty about the balance values in the network with the feedback gained from previous splits. In a simulated environment this method enables nodes to quickly discover channels and paths with enough liquidity to deliver payments of sizes that are close to the total local balance of the sender (given the liquidity actually exists on the network, which according to tests happens in about 95% of all cases).

Thus this method allows for delivering payments that are orders of magnitude larger than what has currently been reported.

## Source Code

Since we did not find any open source min-cost flow solver for a general convex cost function we have implemented them in Scala and Python.

**Due to illness I have to delay the upload of the code but we plan to share the code latest after July 19th** 

## Funding & Future Work

This research was partially funded by the [Norwegian University of Science and Technology](https://en.wikipedia.org/wiki/Norwegian_University_of_Science_and_Technology) 

However as [described in the Paper](https://arxiv.org/abs/2107.05322) there is a lot of work (and probably also research on the way) that needs to be done in order to make this technology deployable on the Lightning Network.  

In particular after [I shared that a breakthrough had been discovered](https://twitter.com/renepickhardt/status/1401514950984712198) by us I had many venture capitalists and commercial parties offering large sums **if** we would find a way to commercalize this in a proprietary way. However I believe that [knowledge should be free and openly available](https://archive.org/stream/GuerillaOpenAccessManifesto/Goamjuly2008_djvu.txt). As Stefan shared this view we decided to take the open path for that particular path finding problem (:

As those offers and opportunity costs exceeded anything that I ever imagined to earn and as funding in academia is always a gamble and the future is unclear I would highly appreciate if you want to help me become an [independent Lightning Network developer and Researcher](https://ln.rene-pickhardt.de) in the future. Thus I will highly appreciate if you consider to support my future work via: 

* Bitcoin / Lightning at: https://donate.ln.rene-pickhardt.de
* Funny money at: https://www.patreon.com/renepickhardt
* Pointing this issue out to Bitcoin OGs or donors!
 
## More on Min-cost Flows
The code in this repository implements the min-cost flow cost scaling algorithm for convex costs presented in the textbook `Network Flows Network Flows: Theory, Algorithms, and Applications` by [Ravindra K. Ahuja](https://en.wikipedia.org/wiki/Ravindra_K._Ahuja), [Thomas L. Magnanti](https://en.wikipedia.org/wiki/Thomas_L._Magnanti) and [James B. Orlin](https://mitmgmtfaculty.mit.edu/jorlin/)

The approach mainly follows chapter 9, 10.2 and 14.5 of the textbook.

Other good resources are the [Lecture series](http://courses.csail.mit.edu/6.854/20/) by [David Karger](http://people.csail.mit.edu/karger/) with [lecture notes](http://courses.csail.mit.edu/6.854/current/Notes/) and more specificially [this one](http://courses.csail.mit.edu/6.854/current/Notes/n09-mincostflow.html). Finally the [relevant subset of the recorded videos can be found at this playlist](https://www.youtube.com/playlist?list=PLaRKlIqjjguDXlnJWG2T7U52iHZl8Edrcv)

Other good resources are the [lecture notes by Dorit Hochbaum](https://hochbaum.ieor.berkeley.edu/)

## Media and Public

* German Speaking Bitcoin Podcast Honigdachs invited us to the show 62 titled [Pickhardt-Payments](https://coinspondent.de/2021/07/11/honigdachs-62-pickhardt-payments/)
* Andreas M. Antonopoulos noted that our method [massively improves the Lightning Network and is a game changer](https://twitter.com/aantonop/status/1403823353366994946).
* The work was presented on [What Bitcoin Did Podcast](https://www.whatbitcoindid.com/podcast/mastering-lightning)
