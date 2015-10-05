#!/bin/sh
exec scala "$0" "$0"
!#
object HelloWord{
	def main(args:Array[String]){
		println("hello world")
	}
}
HelloWord.main(args)