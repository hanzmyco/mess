#!/bin/sh
exec scala "$0" "$0"
!#
object HelloWord{
	def main(args:Array[String]){
		f(3,List(1,2,3,4))

	}
	def f(num:Int,arr:List[Int]):List[Int] = {

		arr.map((i:Int)=>recur(num,i))
		//println arr
		arr
	}
	def recur(num:Int,symbol:Int) : List[Int]={
		val a=List(symbol)
		val temp=List()
		var x=1
		for (x <- 1 to num){
			println(symbol)

			



		}
		List(1)
	}
}
HelloWord.main(args)