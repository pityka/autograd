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
  * The shape of the argument given to that function is the shape of the value of Op (dy/dw2)
  * The shape of the return is the shape of the argument (parameter) with respect the
  * derivative is taken (dy/dw1)
  *
  */
trait Op {
  val value: Variable
  val params: List[(Variable, Mat[Double] => Mat[Double])]
}

case class Variable(
    op: Op,
    value: Mat[Double],
    needsGrad: Boolean = true
) {
  def detach = copy(needsGrad = false)
  def zipBackward(fn: Mat[Double] => Mat[Double]) = (this, fn)

  def accumulateGrad(
      incoming: Mat[Double],
      computeGrad: Mat[Double] => Mat[Double]
  ) = if (needsGrad) {
    val d = computeGrad(incoming)

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

  def +(other: Variable) = Add(this, other).value
  def -(other: Variable) = Minus(this, other).value
  def *(other: Variable) = Mult(this, other).value
  def /(other: Variable) = Div(this, other).value
  def mm(other: Variable) = MatMul(this, other).value
  def relu = Relu(this).value
  def sum = Sum(this).value
  def rowSum = RowSum(this).value
  def colSum = ColSum(this).value
  def exp = Exp(this).value
  def log = Log(this).value
  def sin = Sin(this).value
  def cos = Cos(this).value
  def tan = Tan(this).value
  def atan = ArcTan(this).value
  def pow(const: Double) = PowConst(this, const).value
  def logSoftMax = LogSoftMaxRowWise(this).value
  def crossEntropy(other: Variable) = CrossEntropyRowWise(this, other).value
  def squaredFrobenius = SquaredFrobeniusMatrixNorm(this).value

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

case class Relu(a: Variable) extends ElementwiseOp {

  def op(d: Double) = if (d < 0d) 0d else d
  def diff(d: Double) = if (d < 0d) 0d else 1d

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
case class Minus(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward(p => p),
    b.zipBackward(p => p * (-1d))
  )
  val value = Variable(this, a.value - b.value)

  override def toString = s"(${a.stringify()} - ${b.stringify()})"
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
case class Div(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward(p => {
      p * b.value.map(x => 1d / x)
    }),
    b.zipBackward(p => p * (a.value * b.value.map(x => -1d / (x * x))))
  )

  val value = Variable(this, a.value / b.value)

  override def toString = s"(${a.stringify()} / ${b.stringify()})"
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

trait ElementwiseOp extends Op {
  def a: Variable
  def op(d: Double): Double
  def diff(d: Double): Double

  val params = List(
    a.zipBackward(p => p * a.value.map(diff))
  )
  val value = Variable(this, a.value.map(op))

}

case class Exp(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.exp(d)
  def diff(d: Double) = math.exp(d)

  override def toString = s"EXP(${a.stringify()})"
}
case class Log(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.log(d)
  def diff(d: Double) = 1d / d

  override def toString = s"LOG(${a.stringify()})"
}
case class Sin(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.sin(d)
  def diff(d: Double) = math.cos(d)

  override def toString = s"SIN(${a.stringify()})"
}
case class Cos(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.cos(d)
  def diff(d: Double) = -math.sin(d)

  override def toString = s"COS(${a.stringify()})"
}
case class Tan(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.tan(d)
  def diff(d: Double) = 1 + math.pow(math.tan(d), 2d)

  override def toString = s"COS(${a.stringify()})"
}
case class ArcTan(a: Variable) extends ElementwiseOp {
  def op(d: Double) = math.atan(d)
  def diff(d: Double) = 1d / (1d + d * d)

  override def toString = s"COS(${a.stringify()})"
}

case class PowConst(a: Variable, param: Double) extends ElementwiseOp {
  def op(d: Double) = math.pow(d, param)
  def diff(d: Double) = param * math.pow(d, param - 1d)

  override def toString = s"POW(${a.stringify()},$param)"
}

trait RowWiseOp extends Op {
  def a: Variable
  def op(d: Vec[Double]): Vec[Double]
  def diff(rowIdx: Int): Mat[Double]

  val params = List(
    a.zipBackward { p =>
      p.mapRows { (prow, idx) =>
        val d = diff(idx)
        (Mat(prow) tmm d).row(0)
      }
    }
  )
  val value = Variable(this, a.value.mapRows { (row, _) => op(row) })

}

case class LogSoftMaxRowWise(a: Variable) extends RowWiseOp {

  def diff(rowIdx: Int) = {
    mat.ident(a.value.numCols) + value.value
      .row(Array(rowIdx))
      .map(x => -math.exp(x))
  }

  private def logSumExp(row: Vec[Double]) = {
    val max = row.max2
    math.log(row.map(e => math.exp(e - max)).sum2) + max
  }
  def op(row: Vec[Double]) = {
    val l = logSumExp(row)
    row.map(x => x - l)
  }

  override def toString = s"LOGSOFTMAX(${a.stringify()})"
}

case class CrossEntropyRowWise(a: Variable, b: Variable) extends Op {

  override val params: List[(Variable, Mat[Double] => Mat[Double])] = List(
    a.zipBackward { p => p * (b.value * (-1)) },
    b.zipBackward { p => p * (a.value / b.value) * (-1) }
  )

  val value =
    Variable(
      this,
      Mat(
        a.value.rows
          .zip(b.value.rows)
          .map {
            case (rowa, rowb) =>
              (rowa vv rowb
                .map(math.log)) * -1
          }
          .toVec
      )
    )
  override def toString = s"CROSSENTROPY(${a.stringify()} , ${b.stringify()})"
}

case class SquaredFrobeniusMatrixNorm(a: Variable) extends Op {
  val params = List(
    a.zipBackward { p => p mm (a.value * 2) }
  )
  val value =
    Variable(this, Mat(Vec(a.value.map(x => x * x).toVec.sum2)))
  override def toString = s"FROBENIUS(${a.stringify()})"
}

// each row is a sample, batches are along the first dimension
// https://arxiv.org/pdf/1502.03167.pdf
// case class BatchNorm(a: Variable) extends Op

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
