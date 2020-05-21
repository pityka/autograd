import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import java.{util => ju}

trait Op {
  val value: Variable
  val params: List[(Variable, Option[Mat[Double] => Mat[Double]])]
  // def gradient: List[(Variable, D)]
}

case class Variable(
    op: Op,
    value: Mat[Double]
) {

  /** Zip with a function which calculates the partial derivative of the function value wrt to self
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
    */
  def zipBackward(fn: Mat[Double] => Mat[Double]) = (this, Some(fn))
  def accumulateGrad(
      incoming: Mat[Double],
      computeGrad: Option[Mat[Double] => Mat[Double]]
  ) = computeGrad.foreach { f =>
    val d = f(incoming)

    if (partialDerivative.isEmpty) {
      partialDerivative = Some(d)
    } else {
      partialDerivative = Some(partialDerivative.get + d)
    }
  }
  val id = ju.UUID.randomUUID()
  def stringify(printValue: Boolean = false) =
    if (printValue)
      s"$op == $value"
    else s"$op"
  var partialDerivative: Option[Mat[Double]] = None
}

// val d = v.partialDerivative.get mm gradient

// case class Constant(const: Double) extends Op[Double] {
//   val value = Variable(this, const)
//   def gradient = Nil
//   override def toString = s"$const"
// }

// case class Add(a: Variable[Double], b: Variable[Double]) extends Op[Double] {
//   val value = Variable(this, a.value + b.value)
//   def gradient = List(a -> 1d, b -> 1d)
//   override def toString = s"(${a.stringify()} + ${b.stringify()})"
// }

// case class Mult(a: Variable[Double], b: Variable[Double]) extends Op[Double] {
//   val value = Variable(this, a.value * b.value)
//   def gradient = List(a -> b.value, b -> a.value)
//   override def toString = s"(${a.stringify()} * ${b.stringify()})"
// }

// case class Pow(a: Variable[Double], b: Variable[Double]) extends Op[Double] {
//   val value = Variable(this, math.pow(a.value, b.value))
//   def gradient =
//     List(a -> b.value * math.pow(a.value, b.value - 1))
//   override def toString = s"(${a.stringify()} ^ ${b.stringify()})"
// }

object Autograd {
  // def backprop(v: Variable[Double]): Unit = {
  //   v.partialDerivative = Some(1d)
  //   def loop(v: Variable[Double]): Unit = {
  //     v.op.gradient.filter(_._1.needsGrad).foreach {
  //       case (v1, gradient) =>
  //         val d = v.partialDerivative.get * gradient
  //         if (v1.partialDerivative.isEmpty) {
  //           v1.partialDerivative = Some(0d)
  //         }
  //         v1.partialDerivative = Some(v1.partialDerivative.get + d)
  //         loop(v1)
  //     }
  //   }
  //   loop(v)
  // }
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
    v.partialDerivative = Some(mat.ident(v.value.numRows))
    topologicalSort(v).foreach { v =>
      v.op.params.foreach {
        case (v1, computeGrad) =>
          v1.accumulateGrad(v.partialDerivative.get, computeGrad)

      }
    }

  }
}

case class ConstantVec(const: Mat[Double]) extends Op {
  val params = Nil
  val value = Variable(this, const)
  override def toString = s"$const"
}

case class AddVec(a: Variable, b: Variable) extends Op {
  assert(a.value.numCols == 1)
  assert(b.value.numCols == 1)
  val params = List(
    a.zipBackward(p => p mm mat.ident(a.value.numRows)),
    b.zipBackward(p => p mm mat.ident(a.value.numRows))
  )
  val value = Variable(this, a.value + b.value)

  override def toString = s"(${a.stringify()} + ${b.stringify()})"
}

case class SumVec(a: Variable) extends Op {
  assert(a.value.numCols == 1)
  val params = List(a.zipBackward(p => p mm a.value.T))

  val value = Variable(this, Mat(Vec(a.value.colSums.sum2)))

  override def toString = s"SUM(${a.stringify()})"
}

case class DotVec(a: Variable, b: Variable) extends Op {
  assert(a.value.numCols == 1)
  assert(b.value.numCols == 1)
  val params =
    List(a.zipBackward(p => p mm b.value.T), b.zipBackward(p => p mm a.value.T))

  val value = Variable(this, a.value tmm b.value)

  override def toString = s"(${a.stringify()} dot ${b.stringify()})"
}

case class MultVec(a: Variable, b: Variable) extends Op {
  assert(a.value.numCols == 1)
  assert(b.value.numCols == 1)
  val params = List(
    a.zipBackward(p => p mm mat.diag(b.value.col(0))),
    b.zipBackward(p => p mm mat.diag(a.value.col(0)))
  )

  val value = Variable(this, a.value * b.value)

  override def toString = s"(${a.stringify()} * ${b.stringify()})"
}

object Test extends App {
  val x1 = ConstantVec(Mat(Vec(1d, 2d))).value
  val x2 = ConstantVec(Mat(Vec(3d, 4d))).value
  val tip = AddVec(
    SumVec(MultVec(AddVec(x1, x2).value, x2).value).value,
    DotVec(x1, x2).value
  ).value

  println(tip.stringify(true))
  Autograd.backpropVec(tip)
  println("dx1 " + x1.partialDerivative.get)
  println("dx2 " + x2.partialDerivative.get)

  // val x1_ = Constant(1d).value
  // val x2_ = Constant(3d).value
  // val tip_ = Mult(Add(x1_, x2_).value, x2_).value

  // println(tip_.stringify(true))
  // Autograd.backprop(tip_)
  // println("dx1 " + x1_.partialDerivative.get)
  // println("dx2 " + x2_.partialDerivative.get)
}
