import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import java.{util => ju}

/**
  * Params: the input and the function which calculates the partial derivative
  * of the function value wrt to this input
  *
  * y = f1 o f2 o .. o fn
  *
  * One of these subexpression (f_i) has value w2 and arguments w1.
  * We can write this: dy/dw1 = dy/dw2 * dw2/dw1.
  * dw2/dw1 is the Jacobian of f_i at the current value of w1.
  * dy/dw2 is the Jacobian of y wrt to w2 at the current value of w2.
  *
  * The current value of w1 and w2 are computed in a forward pass.
  * The value dy/dy is 1 and from this dy/dw2 is recursed in the backward pass.
  * The Jacobian function of dw2/dw1 is either computed symbolically.
  *
  * https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
  * http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf
  *
  * The function given in this argument is dy/dw2 => dy/dw2 * dw2/dw1.
  * The argument is coming down from the backward pass.
  * The Op fills in the symbolic part and the multiplication.
  *
  * The shape of the argument given to that function is the transpose of the shape of the value of Op
  * The shape of the return is n x m where n is the value dimension and m is the param dimension
  */
trait Op {
  val value: Variable
  val params: List[(Variable, Option[Mat[Double] => Mat[Double]])]
}

case class Variable(
    op: Op,
    value: Mat[Double],
    needsGrad: Boolean = true
) {
  def detach = copy(needsGrad = false)
  def zipBackward(fn: Mat[Double] => Mat[Double]) = (this, Some(fn))
  def accumulateGrad(
      incoming: Mat[Double],
      computeGrad: Option[Mat[Double] => Mat[Double]]
  ) = if (needsGrad) {
    computeGrad.foreach { f =>
      val d = f(incoming)

      if (partialDerivative.isEmpty) {
        partialDerivative = Some(d)
      } else {
        partialDerivative = Some(partialDerivative.get + d)
      }
    }
  }
  val id = ju.UUID.randomUUID()
  def stringify(printValue: Boolean = false) =
    if (printValue)
      s"$op == $value"
    else s"$op"
  var partialDerivative: Option[Mat[Double]] = None

  def +(other: Variable) = Add(this, other).value
  def -(other: Variable) =
    Add(this, Mult(other, Constant(Mat(Vec(-1d))).value).value).value

  def *(other: Variable) = Mult(this, other).value
  def mm(other: Variable) = MatMul(this, other).value
  def relu = Relu(this).value
  def sum = Sum(this).value
  def rowSum = RowSum(this).value

}

object Autograd {

  private def topologicalSort[D](root: Variable): Seq[Variable] = {
    type V = Variable
    var order = List.empty[V]
    var marks = Set.empty[ju.UUID]
    var currentParents = Set.empty[ju.UUID]

    def visit(n: V): Unit =
      if (marks.contains(n.id)) ()
      else {
        if (currentParents.contains(n.id)) {
          println(s"error: loop to ${n.id}")
          ()
        } else {
          currentParents = currentParents + n.id
          val children = n.op.params.map(_._1)
          children.foreach(visit)
          currentParents = currentParents - n.id
          marks = marks + n.id
          order = n :: order
        }
      }

    visit(root)

    order

  }
  def backpropVec(v: Variable): Unit = {
    v.partialDerivative = Some(mat.ones(v.value.numRows, v.value.numCols))
    topologicalSort(v).foreach { v =>
      v.op.params.foreach {
        case (v1, computeGrad) =>
          v1.accumulateGrad(v.partialDerivative.get, computeGrad)

      }
    }

  }
}

case class Relu(a: Variable) extends Op {
  val params = List(
    a.zipBackward(p => p.map(x => if (x < 0) 0d else x))
  )
  val value = Variable(this, a.value.map(x => if (x < 0) 0d else x))

  override def toString = s"RELU(${a.stringify()})"
}

case class Constant(const: Mat[Double]) extends Op {
  val params = Nil
  val value = Variable(this, const)
  override def toString = s"$const"
}

case class Add(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward(p => p),
    b.zipBackward(p => p)
  )
  val value = Variable(this, a.value + b.value)

  override def toString = s"(${a.stringify()} + ${b.stringify()})"
}

case class Mult(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward(p => {
      p * b.value
    }),
    b.zipBackward(p => p * a.value)
  )

  val value = Variable(this, a.value * b.value)

  override def toString = s"(${a.stringify()} * ${b.stringify()})"
}

case class Sum(a: Variable) extends Op {
  val params = List(a.zipBackward(p => {
    (p mm mat.ones(1, a.value.length))
  }))

  val value = Variable(this, Mat(Vec(a.value.colSums.sum2)))

  override def toString = s"SUM(${a.stringify()})"
}
case class ColSum(a: Variable) extends Op {
  val params = List(a.zipBackward(p => {
    (p * a.value.numRows)
  }))

  val value = Variable(this, Mat(a.value.colSums).T)

  override def toString = s"COLSUM(${a.stringify()})"
}
case class RowSum(a: Variable) extends Op {
  val params = List(a.zipBackward(p => {
    (p * a.value.numCols)
  }))

  val value = Variable(this, Mat(a.value.rowSums))

  override def toString = s"ROWSUM(${a.stringify()})"
}

// http://cs231n.stanford.edu/handouts/derivatives.pdf
case class MatMul(a: Variable, b: Variable) extends Op {
  val params =
    List(a.zipBackward(p => p mmt b.value), b.zipBackward(p => a.value tmm p))

  val value = Variable(this, a.value mm b.value)

  override def toString = s"(${a.stringify()} dot ${b.stringify()})"
}

object Test extends App {
  val x1 = Constant(Mat(Vec(1d, -20d, 3d), Vec(3d, 4d, 3d))).value
  val x2 = Constant(Mat(Vec(3d, 4d, 3d), Vec(5d, 6d, 3d)).T).value
  val x3 = Constant(Mat(Vec(3d))).value
  val tip = ColSum(
    RowSum(Relu(Mult(MatMul(x1, x2).value, MatMul(x1, x2).value).value).value).value
  ).value

  val tip2 = (x1 + x3) mm x2

  println(tip2.stringify(true))
  Autograd.backpropVec(tip2)
  println("dx1 " + x1.partialDerivative.get)
  println("dx2 " + x2.partialDerivative.get)
  // println("dx3 " + x3.partialDerivative.get)

  // val x1_ = Constant(1d).value
  // val x2_ = Constant(3d).value
  // val tip_ = Mult(Add(x1_, x2_).value, x2_).value

  // println(tip_.stringify(true))
  // Autograd.backprop(tip_)
  // println("dx1 " + x1_.partialDerivative.get)
  // println("dx2 " + x2_.partialDerivative.get)
}
