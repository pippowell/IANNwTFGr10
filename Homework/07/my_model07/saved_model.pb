ĥ)
?#?#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements(
handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.10.02unknown8Ə&
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
Adam/rnn/ourlstm/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_3/bias/v
?
3Adam/rnn/ourlstm/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_3/bias/v*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_3/kernel/v
?
5Adam/rnn/ourlstm/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_3/kernel/v*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_2/bias/v
?
3Adam/rnn/ourlstm/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_2/bias/v*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_2/kernel/v
?
5Adam/rnn/ourlstm/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_2/kernel/v*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_1/bias/v
?
3Adam/rnn/ourlstm/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_1/bias/v*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_1/kernel/v
?
5Adam/rnn/ourlstm/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_1/kernel/v*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/rnn/ourlstm/dense/bias/v
?
1Adam/rnn/ourlstm/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/rnn/ourlstm/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*0
shared_name!Adam/rnn/ourlstm/dense/kernel/v
?
3Adam/rnn/ourlstm/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense/kernel/v*
_output_shapes

:8*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:0*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:00*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:00*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:0*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
Adam/rnn/ourlstm/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_3/bias/m
?
3Adam/rnn/ourlstm/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_3/bias/m*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_3/kernel/m
?
5Adam/rnn/ourlstm/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_3/kernel/m*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_2/bias/m
?
3Adam/rnn/ourlstm/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_2/bias/m*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_2/kernel/m
?
5Adam/rnn/ourlstm/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_2/kernel/m*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/rnn/ourlstm/dense_1/bias/m
?
3Adam/rnn/ourlstm/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense_1/bias/m*
_output_shapes
:*
dtype0
?
!Adam/rnn/ourlstm/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*2
shared_name#!Adam/rnn/ourlstm/dense_1/kernel/m
?
5Adam/rnn/ourlstm/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/rnn/ourlstm/dense_1/kernel/m*
_output_shapes

:8*
dtype0
?
Adam/rnn/ourlstm/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/rnn/ourlstm/dense/bias/m
?
1Adam/rnn/ourlstm/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/rnn/ourlstm/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*0
shared_name!Adam/rnn/ourlstm/dense/kernel/m
?
3Adam/rnn/ourlstm/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn/ourlstm/dense/kernel/m*
_output_shapes

:8*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:0*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:00*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:00*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:0*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
rnn/ourlstm/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namernn/ourlstm/dense_3/bias
?
,rnn/ourlstm/dense_3/bias/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_3/bias*
_output_shapes
:*
dtype0
?
rnn/ourlstm/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*+
shared_namernn/ourlstm/dense_3/kernel
?
.rnn/ourlstm/dense_3/kernel/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_3/kernel*
_output_shapes

:8*
dtype0
?
rnn/ourlstm/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namernn/ourlstm/dense_2/bias
?
,rnn/ourlstm/dense_2/bias/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_2/bias*
_output_shapes
:*
dtype0
?
rnn/ourlstm/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*+
shared_namernn/ourlstm/dense_2/kernel
?
.rnn/ourlstm/dense_2/kernel/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_2/kernel*
_output_shapes

:8*
dtype0
?
rnn/ourlstm/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namernn/ourlstm/dense_1/bias
?
,rnn/ourlstm/dense_1/bias/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_1/bias*
_output_shapes
:*
dtype0
?
rnn/ourlstm/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*+
shared_namernn/ourlstm/dense_1/kernel
?
.rnn/ourlstm/dense_1/kernel/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense_1/kernel*
_output_shapes

:8*
dtype0
?
rnn/ourlstm/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namernn/ourlstm/dense/bias
}
*rnn/ourlstm/dense/bias/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense/bias*
_output_shapes
:*
dtype0
?
rnn/ourlstm/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*)
shared_namernn/ourlstm/dense/kernel
?
,rnn/ourlstm/dense/kernel/Read/ReadVariableOpReadVariableOprnn/ourlstm/dense/kernel*
_output_shapes

:8*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:0*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:0*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:0*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:0*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:0*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:00*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:0*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:00*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:0*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:0*
dtype0
?
serving_default_input_1Placeholder*<
_output_shapes*
(:&??????????????????*
dtype0*1
shape(:&??????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancernn/ourlstm/dense/kernelrnn/ourlstm/dense/biasrnn/ourlstm/dense_1/kernelrnn/ourlstm/dense_1/biasrnn/ourlstm/dense_2/kernelrnn/ourlstm/dense_2/biasrnn/ourlstm/dense_3/kernelrnn/ourlstm/dense_3/bias!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_1/betabatch_normalization_1/gammadense_4/kerneldense_4/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_25830

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ގ
valueӎBώ Bǎ
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

convlayer1
	
convlayer2


convlayer3

batchnorm1
global_pool
timedist
rnn

batchnorm2
outputlayer
	optimizer
call

signatures*
?
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23*
?
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
*18
+19*
* 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias
 ?_jit_compiled_convolution_op*
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

kernel
bias
 F_jit_compiled_convolution_op*
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op*
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
moving_variance*
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
	layer* 
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
gcell
h
state_spec*
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oaxis
	&gamma
'beta
(moving_mean
)moving_variance*
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

*kernel
+bias*
?
viter

wbeta_1

xbeta_2
	ydecay
zlearning_ratem?m?m?m?m?m?m?m?m?m? m?!m?"m?#m?$m?%m?&m?'m?*m?+m?v?v?v?v?v?v?v?v?v?v? v?!v?"v?#v?$v?%v?&v?'v?*v?+v?*

{trace_0* 

|serving_default* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUErnn/ourlstm/dense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUErnn/ourlstm/dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn/ourlstm/dense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUErnn/ourlstm/dense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn/ourlstm/dense_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUErnn/ourlstm/dense_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn/ourlstm/dense_3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUErnn/ourlstm/dense_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_1/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
 
0
1
(2
)3*
C
0
	1

2
3
4
5
6
7
8*

}0
~1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
 
0
1
2
3*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
<
0
1
 2
!3
"4
#5
$6
%7*
<
0
1
 2
!3
"4
#5
$6
%7*
* 
?
?states
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
forgetgate
?
inputgate1
?
inputgate2
?
outputgate*
* 
 
&0
'1
(2
)3*

&0
'1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

*0
+1*

*0
+1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

g0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
0
1
 2
!3
"4
#5
$6
%7*
<
0
1
 2
!3
"4
#5
$6
%7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

 kernel
!bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

"kernel
#bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

$kernel
%bias*

(0
)1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

 0
!1*

 0
!1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

"0
#1*

"0
#1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/batch_normalization/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/batch_normalization/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/rnn/ourlstm/dense/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_2/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_2/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_3/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_3/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/batch_normalization/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/batch_normalization/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/rnn/ourlstm/dense/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_2/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_2/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn/ourlstm/dense_3/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/rnn/ourlstm/dense_3/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp,rnn/ourlstm/dense/kernel/Read/ReadVariableOp*rnn/ourlstm/dense/bias/Read/ReadVariableOp.rnn/ourlstm/dense_1/kernel/Read/ReadVariableOp,rnn/ourlstm/dense_1/bias/Read/ReadVariableOp.rnn/ourlstm/dense_2/kernel/Read/ReadVariableOp,rnn/ourlstm/dense_2/bias/Read/ReadVariableOp.rnn/ourlstm/dense_3/kernel/Read/ReadVariableOp,rnn/ourlstm/dense_3/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp3Adam/rnn/ourlstm/dense/kernel/m/Read/ReadVariableOp1Adam/rnn/ourlstm/dense/bias/m/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_1/kernel/m/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_1/bias/m/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_2/kernel/m/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_2/bias/m/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_3/kernel/m/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp3Adam/rnn/ourlstm/dense/kernel/v/Read/ReadVariableOp1Adam/rnn/ourlstm/dense/bias/v/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_1/kernel/v/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_1/bias/v/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_2/kernel/v/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_2/bias/v/Read/ReadVariableOp5Adam/rnn/ourlstm/dense_3/kernel/v/Read/ReadVariableOp3Adam/rnn/ourlstm/dense_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_28068
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancernn/ourlstm/dense/kernelrnn/ourlstm/dense/biasrnn/ourlstm/dense_1/kernelrnn/ourlstm/dense_1/biasrnn/ourlstm/dense_2/kernelrnn/ourlstm/dense_2/biasrnn/ourlstm/dense_3/kernelrnn/ourlstm/dense_3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/rnn/ourlstm/dense/kernel/mAdam/rnn/ourlstm/dense/bias/m!Adam/rnn/ourlstm/dense_1/kernel/mAdam/rnn/ourlstm/dense_1/bias/m!Adam/rnn/ourlstm/dense_2/kernel/mAdam/rnn/ourlstm/dense_2/bias/m!Adam/rnn/ourlstm/dense_3/kernel/mAdam/rnn/ourlstm/dense_3/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/rnn/ourlstm/dense/kernel/vAdam/rnn/ourlstm/dense/bias/v!Adam/rnn/ourlstm/dense_1/kernel/vAdam/rnn/ourlstm/dense_1/bias/v!Adam/rnn/ourlstm/dense_2/kernel/vAdam/rnn/ourlstm/dense_2/bias/v!Adam/rnn/ourlstm/dense_3/kernel/vAdam/rnn/ourlstm/dense_3/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_28297׭#
?
?
'__inference_ourlstm_layer_call_fn_27783

inputs
states_0
states_1
unknown:8
	unknown_0:
	unknown_1:8
	unknown_2:
	unknown_3:8
	unknown_4:
	unknown_5:8
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????0:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
B__inference_dense_4_layer_call_and_return_conditional_losses_27756

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_26665

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?%
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491

inputs?
%conv2d_conv2d_readvariableop_resource:00@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?

?
while_cond_27532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_27532___redundant_placeholder03
/while_while_cond_27532___redundant_placeholder13
/while_while_cond_27532___redundant_placeholder23
/while_while_cond_27532___redundant_placeholder33
/while_while_cond_27532___redundant_placeholder43
/while_while_cond_27532___redundant_placeholder53
/while_while_cond_27532___redundant_placeholder63
/while_while_cond_27532___redundant_placeholder73
/while_while_cond_27532___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?1
?	
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_24785
x&
conv2d_24453:0
conv2d_24455:0(
conv2d_1_24492:00
conv2d_1_24494:0(
conv2d_2_24531:00
conv2d_2_24533:0'
batch_normalization_24536:0'
batch_normalization_24538:0'
batch_normalization_24540:0'
batch_normalization_24542:0
	rnn_24722:8
	rnn_24724:
	rnn_24726:8
	rnn_24728:
	rnn_24730:8
	rnn_24732:
	rnn_24734:8
	rnn_24736:)
batch_normalization_1_24739:)
batch_normalization_1_24741:)
batch_normalization_1_24743:)
batch_normalization_1_24745:
dense_4_24779:
dense_4_24781:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_24453conv2d_24455*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_24452?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_24492conv2d_1_24494*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_24531conv2d_2_24533*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_24536batch_normalization_24538batch_normalization_24540batch_normalization_24542*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23831?
 time_distributed/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23906w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape4batch_normalization/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_24722	rnn_24724	rnn_24726	rnn_24728	rnn_24730	rnn_24732	rnn_24734	rnn_24736*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_24721?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0batch_normalization_1_24739batch_normalization_1_24741batch_normalization_1_24743batch_normalization_1_24745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24354?
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_4_24779dense_4_24781*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_24778?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?9
?
>__inference_rnn_layer_call_and_return_conditional_losses_24120

inputs
ourlstm_24009:8
ourlstm_24011:
ourlstm_24013:8
ourlstm_24015:
ourlstm_24017:8
ourlstm_24019:
ourlstm_24021:8
ourlstm_24023:
identity??ourlstm/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask?
ourlstm/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ourlstm_24009ourlstm_24011ourlstm_24013ourlstm_24015ourlstm_24017ourlstm_24019ourlstm_24021ourlstm_24023*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ourlstm_24009ourlstm_24011ourlstm_24013ourlstm_24015ourlstm_24017ourlstm_24019ourlstm_24021ourlstm_24023*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_24032*
condR
while_cond_24031*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p
NoOpNoOp ^ourlstm/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2B
ourlstm/StatefulPartitionedCallourlstm/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?
?
(__inference_conv2d_2_layer_call_fn_26707

inputs!
unknown:00
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
??
?/
!__inference__traced_restore_28297
file_prefix8
assignvariableop_conv2d_kernel:0,
assignvariableop_1_conv2d_bias:0<
"assignvariableop_2_conv2d_1_kernel:00.
 assignvariableop_3_conv2d_1_bias:0<
"assignvariableop_4_conv2d_2_kernel:00.
 assignvariableop_5_conv2d_2_bias:0:
,assignvariableop_6_batch_normalization_gamma:09
+assignvariableop_7_batch_normalization_beta:0@
2assignvariableop_8_batch_normalization_moving_mean:0D
6assignvariableop_9_batch_normalization_moving_variance:0>
,assignvariableop_10_rnn_ourlstm_dense_kernel:88
*assignvariableop_11_rnn_ourlstm_dense_bias:@
.assignvariableop_12_rnn_ourlstm_dense_1_kernel:8:
,assignvariableop_13_rnn_ourlstm_dense_1_bias:@
.assignvariableop_14_rnn_ourlstm_dense_2_kernel:8:
,assignvariableop_15_rnn_ourlstm_dense_2_bias:@
.assignvariableop_16_rnn_ourlstm_dense_3_kernel:8:
,assignvariableop_17_rnn_ourlstm_dense_3_bias:=
/assignvariableop_18_batch_normalization_1_gamma:<
.assignvariableop_19_batch_normalization_1_beta:C
5assignvariableop_20_batch_normalization_1_moving_mean:G
9assignvariableop_21_batch_normalization_1_moving_variance:4
"assignvariableop_22_dense_4_kernel:.
 assignvariableop_23_dense_4_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: #
assignvariableop_31_total: #
assignvariableop_32_count: B
(assignvariableop_33_adam_conv2d_kernel_m:04
&assignvariableop_34_adam_conv2d_bias_m:0D
*assignvariableop_35_adam_conv2d_1_kernel_m:006
(assignvariableop_36_adam_conv2d_1_bias_m:0D
*assignvariableop_37_adam_conv2d_2_kernel_m:006
(assignvariableop_38_adam_conv2d_2_bias_m:0B
4assignvariableop_39_adam_batch_normalization_gamma_m:0A
3assignvariableop_40_adam_batch_normalization_beta_m:0E
3assignvariableop_41_adam_rnn_ourlstm_dense_kernel_m:8?
1assignvariableop_42_adam_rnn_ourlstm_dense_bias_m:G
5assignvariableop_43_adam_rnn_ourlstm_dense_1_kernel_m:8A
3assignvariableop_44_adam_rnn_ourlstm_dense_1_bias_m:G
5assignvariableop_45_adam_rnn_ourlstm_dense_2_kernel_m:8A
3assignvariableop_46_adam_rnn_ourlstm_dense_2_bias_m:G
5assignvariableop_47_adam_rnn_ourlstm_dense_3_kernel_m:8A
3assignvariableop_48_adam_rnn_ourlstm_dense_3_bias_m:D
6assignvariableop_49_adam_batch_normalization_1_gamma_m:C
5assignvariableop_50_adam_batch_normalization_1_beta_m:;
)assignvariableop_51_adam_dense_4_kernel_m:5
'assignvariableop_52_adam_dense_4_bias_m:B
(assignvariableop_53_adam_conv2d_kernel_v:04
&assignvariableop_54_adam_conv2d_bias_v:0D
*assignvariableop_55_adam_conv2d_1_kernel_v:006
(assignvariableop_56_adam_conv2d_1_bias_v:0D
*assignvariableop_57_adam_conv2d_2_kernel_v:006
(assignvariableop_58_adam_conv2d_2_bias_v:0B
4assignvariableop_59_adam_batch_normalization_gamma_v:0A
3assignvariableop_60_adam_batch_normalization_beta_v:0E
3assignvariableop_61_adam_rnn_ourlstm_dense_kernel_v:8?
1assignvariableop_62_adam_rnn_ourlstm_dense_bias_v:G
5assignvariableop_63_adam_rnn_ourlstm_dense_1_kernel_v:8A
3assignvariableop_64_adam_rnn_ourlstm_dense_1_bias_v:G
5assignvariableop_65_adam_rnn_ourlstm_dense_2_kernel_v:8A
3assignvariableop_66_adam_rnn_ourlstm_dense_2_bias_v:G
5assignvariableop_67_adam_rnn_ourlstm_dense_3_kernel_v:8A
3assignvariableop_68_adam_rnn_ourlstm_dense_3_bias_v:D
6assignvariableop_69_adam_batch_normalization_1_gamma_v:C
5assignvariableop_70_adam_batch_normalization_1_beta_v:;
)assignvariableop_71_adam_dense_4_kernel_v:5
'assignvariableop_72_adam_dense_4_bias_v:
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*? 
value? B? JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp,assignvariableop_10_rnn_ourlstm_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_rnn_ourlstm_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_rnn_ourlstm_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_rnn_ourlstm_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_rnn_ourlstm_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_rnn_ourlstm_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rnn_ourlstm_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_rnn_ourlstm_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_1_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_1_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_1_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_1_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_batch_normalization_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_batch_normalization_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_rnn_ourlstm_dense_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_rnn_ourlstm_dense_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adam_rnn_ourlstm_dense_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_rnn_ourlstm_dense_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adam_rnn_ourlstm_dense_2_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_rnn_ourlstm_dense_2_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_rnn_ourlstm_dense_3_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_rnn_ourlstm_dense_3_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_1_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_1_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_4_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_4_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv2d_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_batch_normalization_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_batch_normalization_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp3assignvariableop_61_adam_rnn_ourlstm_dense_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp1assignvariableop_62_adam_rnn_ourlstm_dense_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp5assignvariableop_63_adam_rnn_ourlstm_dense_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp3assignvariableop_64_adam_rnn_ourlstm_dense_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_rnn_ourlstm_dense_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp3assignvariableop_66_adam_rnn_ourlstm_dense_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp5assignvariableop_67_adam_rnn_ourlstm_dense_3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp3assignvariableop_68_adam_rnn_ourlstm_dense_3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_1_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_1_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_dense_4_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_dense_4_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_27637

inputs>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_27533*
condR
while_cond_27532*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?%
?
A__inference_conv2d_layer_call_and_return_conditional_losses_26656

inputs?
%conv2d_conv2d_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_27650

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24354|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_27663

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24401|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_4_layer_call_and_return_conditional_losses_24778

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26802

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
?

?
while_cond_24031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24031___redundant_placeholder03
/while_while_cond_24031___redundant_placeholder13
/while_while_cond_24031___redundant_placeholder23
/while_while_cond_24031___redundant_placeholder33
/while_while_cond_24031___redundant_placeholder43
/while_while_cond_24031___redundant_placeholder53
/while_while_cond_24031___redundant_placeholder63
/while_while_cond_24031___redundant_placeholder73
/while_while_cond_24031___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
3__inference_batch_normalization_layer_call_fn_26753

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23831?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
??
?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26614
xF
,conv2d_conv2d_conv2d_readvariableop_resource:0G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_1_conv2d_conv2d_readvariableop_resource:00I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_2_conv2d_conv2d_readvariableop_resource:00I
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:09
+batch_normalization_readvariableop_resource:0;
-batch_normalization_readvariableop_1_resource:0J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:0L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:0B
0rnn_ourlstm_dense_matmul_readvariableop_resource:8?
1rnn_ourlstm_dense_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_1_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_1_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_2_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_2_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_3_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_3_biasadd_readvariableop_resource:K
=batch_normalization_1_assignmovingavg_readvariableop_resource:M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?#conv2d/Conv2D/Conv2D/ReadVariableOp?0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_1/Conv2D/Conv2D/ReadVariableOp?2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_2/Conv2D/Conv2D/ReadVariableOp?2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?(rnn/ourlstm/dense/BiasAdd/ReadVariableOp?'rnn/ourlstm/dense/MatMul/ReadVariableOp?*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_1/MatMul/ReadVariableOp?*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_2/MatMul/ReadVariableOp?*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_3/MatMul/ReadVariableOp?	rnn/whileD
conv2d/Conv2D/ShapeShapex*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
conv2d/Conv2D/ReshapeReshapex$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0`
conv2d_2/Conv2D/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_2/Conv2D/ReshapeReshapeconv2d_1/Relu:activations:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*X
_output_shapesF
D:&??????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(n
time_distributed/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0m
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????d
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/Reshape_2Reshape(batch_normalization/FusedBatchNormV3:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????0\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_maskb
rnn/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/ourlstm/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros_1:output:0 rnn/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
'rnn/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp0rnn_ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense/MatMulMatMulrnn/ourlstm/concat:output:0/rnn/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(rnn/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp1rnn_ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense/BiasAddBiasAdd"rnn/ourlstm/dense/MatMul:product:00rnn/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
rnn/ourlstm/dense/SigmoidSigmoid"rnn/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
rnn/ourlstm/MulMulrnn/ourlstm/dense/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_1/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_1/BiasAddBiasAdd$rnn/ourlstm/dense_1/MatMul:product:02rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_1/SigmoidSigmoid$rnn/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_2/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_2/BiasAddBiasAdd$rnn/ourlstm/dense_2/MatMul:product:02rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
rnn/ourlstm/dense_2/TanhTanh$rnn/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_1Mulrnn/ourlstm/dense_1/Sigmoid:y:0rnn/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????v
rnn/ourlstm/AddAddV2rnn/ourlstm/Mul:z:0rnn/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_3/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_3/BiasAddBiasAdd$rnn/ourlstm/dense_3/MatMul:product:02rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_3/SigmoidSigmoid$rnn/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
rnn/ourlstm/TanhTanhrnn/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_2Mulrnn/ourlstm/dense_3/Sigmoid:y:0rnn/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:00rnn_ourlstm_dense_matmul_readvariableop_resource1rnn_ourlstm_dense_biasadd_readvariableop_resource2rnn_ourlstm_dense_1_matmul_readvariableop_resource3rnn_ourlstm_dense_1_biasadd_readvariableop_resource2rnn_ourlstm_dense_2_matmul_readvariableop_resource3rnn_ourlstm_dense_2_biasadd_readvariableop_resource2rnn_ourlstm_dense_3_matmul_readvariableop_resource3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_26454* 
condR
rnn_while_cond_26453*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
"batch_normalization_1/moments/meanMeanrnn/transpose_1:y:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencernn/transpose_1:y:03batch_normalization_1/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :???????????????????
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/mul_1Mulrnn/transpose_1:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :???????????????????
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :???????????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_4/Tensordot/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0!dense_4/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????t
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????

NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp)^rnn/ourlstm/dense/BiasAdd/ReadVariableOp(^rnn/ourlstm/dense/MatMul/ReadVariableOp+^rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_1/MatMul/ReadVariableOp+^rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_2/MatMul/ReadVariableOp+^rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2T
(rnn/ourlstm/dense/BiasAdd/ReadVariableOp(rnn/ourlstm/dense/BiasAdd/ReadVariableOp2R
'rnn/ourlstm/dense/MatMul/ReadVariableOp'rnn/ourlstm/dense/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_1/MatMul/ReadVariableOp)rnn/ourlstm/dense_1/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_2/MatMul/ReadVariableOp)rnn/ourlstm/dense_2/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_3/MatMul/ReadVariableOp)rnn/ourlstm/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?%
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530

inputs?
%conv2d_conv2d_readvariableop_resource:00@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
?
rnn_while_cond_26453$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_26453___redundant_placeholder0;
7rnn_while_rnn_while_cond_26453___redundant_placeholder1;
7rnn_while_rnn_while_cond_26453___redundant_placeholder2;
7rnn_while_rnn_while_cond_26453___redundant_placeholder3;
7rnn_while_rnn_while_cond_26453___redundant_placeholder4;
7rnn_while_rnn_while_cond_26453___redundant_placeholder5;
7rnn_while_rnn_while_cond_26453___redundant_placeholder6;
7rnn_while_rnn_while_cond_26453___redundant_placeholder7;
7rnn_while_rnn_while_cond_26453___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?1
?	
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25437
input_1&
conv2d_25378:0
conv2d_25380:0(
conv2d_1_25383:00
conv2d_1_25385:0(
conv2d_2_25388:00
conv2d_2_25390:0'
batch_normalization_25393:0'
batch_normalization_25395:0'
batch_normalization_25397:0'
batch_normalization_25399:0
	rnn_25405:8
	rnn_25407:
	rnn_25409:8
	rnn_25411:
	rnn_25413:8
	rnn_25415:
	rnn_25417:8
	rnn_25419:)
batch_normalization_1_25422:)
batch_normalization_1_25424:)
batch_normalization_1_25426:)
batch_normalization_1_25428:
dense_4_25431:
dense_4_25433:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_25378conv2d_25380*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_24452?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_25383conv2d_1_25385*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_25388conv2d_2_25390*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_25393batch_normalization_25395batch_normalization_25397batch_normalization_25399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23862?
 time_distributed/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23927w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape4batch_normalization/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_25405	rnn_25407	rnn_25409	rnn_25411	rnn_25413	rnn_25415	rnn_25417	rnn_25419*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_25043?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0batch_normalization_1_25422batch_normalization_1_25424batch_normalization_1_25426batch_normalization_1_25428*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24401?
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_4_25431dense_4_25433*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_24778?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_26857

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
global_average_pooling2d/MeanMeanReshape:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape&global_average_pooling2d/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24354

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?,
?
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008

inputs

states
states_16
$dense_matmul_readvariableop_resource:83
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:85
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:85
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:85
'dense_3_biasadd_readvariableop_resource:
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????u
concatConcatV2inputsstates_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0~
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
MulMuldense/Sigmoid:y:0states*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_1/MatMulMatMulconcat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_2/MatMulMatMulconcat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
Mul_1Muldense_1/Sigmoid:y:0dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????R
AddAddV2Mul:z:0	Mul_1:z:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????G
TanhTanhAdd:z:0*
T0*'
_output_shapes
:?????????]
Mul_2Muldense_3/Sigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	Mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	Mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????X

Identity_2IdentityAdd:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????0:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?9
?
>__inference_rnn_layer_call_and_return_conditional_losses_24311

inputs
ourlstm_24200:8
ourlstm_24202:
ourlstm_24204:8
ourlstm_24206:
ourlstm_24208:8
ourlstm_24210:
ourlstm_24212:8
ourlstm_24214:
identity??ourlstm/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask?
ourlstm/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ourlstm_24200ourlstm_24202ourlstm_24204ourlstm_24206ourlstm_24208ourlstm_24210ourlstm_24212ourlstm_24214*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ourlstm_24200ourlstm_24202ourlstm_24204ourlstm_24206ourlstm_24208ourlstm_24210ourlstm_24212ourlstm_24214*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_24223*
condR
while_cond_24222*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p
NoOpNoOp ^ourlstm/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2B
ourlstm/StatefulPartitionedCallourlstm/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?	
?
#__inference_rnn_layer_call_fn_26899
inputs_0
unknown:8
	unknown_0:
	unknown_1:8
	unknown_2:
	unknown_3:8
	unknown_4:
	unknown_5:8
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_24311|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?

?
while_cond_27184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_27184___redundant_placeholder03
/while_while_cond_27184___redundant_placeholder13
/while_while_cond_27184___redundant_placeholder23
/while_while_cond_27184___redundant_placeholder33
/while_while_cond_27184___redundant_placeholder43
/while_while_cond_27184___redundant_placeholder53
/while_while_cond_27184___redundant_placeholder63
/while_while_cond_27184___redundant_placeholder73
/while_while_cond_27184___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
T
8__inference_global_average_pooling2d_layer_call_fn_26807

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23883i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?1
?	
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25209
x&
conv2d_25150:0
conv2d_25152:0(
conv2d_1_25155:00
conv2d_1_25157:0(
conv2d_2_25160:00
conv2d_2_25162:0'
batch_normalization_25165:0'
batch_normalization_25167:0'
batch_normalization_25169:0'
batch_normalization_25171:0
	rnn_25177:8
	rnn_25179:
	rnn_25181:8
	rnn_25183:
	rnn_25185:8
	rnn_25187:
	rnn_25189:8
	rnn_25191:)
batch_normalization_1_25194:)
batch_normalization_1_25196:)
batch_normalization_1_25198:)
batch_normalization_1_25200:
dense_4_25203:
dense_4_25205:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_25150conv2d_25152*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_24452?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_25155conv2d_1_25157*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_25160conv2d_2_25162*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_25165batch_normalization_25167batch_normalization_25169batch_normalization_25171*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23862?
 time_distributed/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23927w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape4batch_normalization/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_25177	rnn_25179	rnn_25181	rnn_25183	rnn_25185	rnn_25187	rnn_25189	rnn_25191*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_25043?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0batch_normalization_1_25194batch_normalization_1_25196batch_normalization_1_25198batch_normalization_1_25200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24401?
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_4_25203dense_4_25205*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_24778?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?U
?
while_body_27533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

?
while_cond_24222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24222___redundant_placeholder03
/while_while_cond_24222___redundant_placeholder13
/while_while_cond_24222___redundant_placeholder23
/while_while_cond_24222___redundant_placeholder33
/while_while_cond_24222___redundant_placeholder43
/while_while_cond_24222___redundant_placeholder53
/while_while_cond_24222___redundant_placeholder63
/while_while_cond_24222___redundant_placeholder73
/while_while_cond_24222___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?U
?
while_body_24939
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?	
?
#__inference_rnn_layer_call_fn_26941

inputs
unknown:8
	unknown_0:
	unknown_1:8
	unknown_2:
	unknown_3:8
	unknown_4:
	unknown_5:8
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_25043|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?

?
while_cond_24616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24616___redundant_placeholder03
/while_while_cond_24616___redundant_placeholder13
/while_while_cond_24616___redundant_placeholder23
/while_while_cond_24616___redundant_placeholder33
/while_while_cond_24616___redundant_placeholder43
/while_while_cond_24616___redundant_placeholder53
/while_while_cond_24616___redundant_placeholder63
/while_while_cond_24616___redundant_placeholder73
/while_while_cond_24616___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_26813

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_27726

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_24778|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
rnn_while_cond_26121$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_26121___redundant_placeholder0;
7rnn_while_rnn_while_cond_26121___redundant_placeholder1;
7rnn_while_rnn_while_cond_26121___redundant_placeholder2;
7rnn_while_rnn_while_cond_26121___redundant_placeholder3;
7rnn_while_rnn_while_cond_26121___redundant_placeholder4;
7rnn_while_rnn_while_cond_26121___redundant_placeholder5;
7rnn_while_rnn_while_cond_26121___redundant_placeholder6;
7rnn_while_rnn_while_cond_26121___redundant_placeholder7;
7rnn_while_rnn_while_cond_26121___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?

?
while_cond_27010
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_27010___redundant_placeholder03
/while_while_cond_27010___redundant_placeholder13
/while_while_cond_27010___redundant_placeholder23
/while_while_cond_27010___redundant_placeholder33
/while_while_cond_27010___redundant_placeholder43
/while_while_cond_27010___redundant_placeholder53
/while_while_cond_27010___redundant_placeholder63
/while_while_cond_27010___redundant_placeholder73
/while_while_cond_27010___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
__inference_call_23758
xF
,conv2d_conv2d_conv2d_readvariableop_resource:0G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_1_conv2d_conv2d_readvariableop_resource:00I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_2_conv2d_conv2d_readvariableop_resource:00I
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:09
+batch_normalization_readvariableop_resource:0;
-batch_normalization_readvariableop_1_resource:0J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:0L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:0B
0rnn_ourlstm_dense_matmul_readvariableop_resource:8?
1rnn_ourlstm_dense_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_1_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_1_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_2_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_2_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_3_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_3_biasadd_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:B
4batch_normalization_1_cast_2_readvariableop_resource:B
4batch_normalization_1_cast_3_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?#conv2d/Conv2D/Conv2D/ReadVariableOp?0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_1/Conv2D/Conv2D/ReadVariableOp?2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_2/Conv2D/Conv2D/ReadVariableOp?2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?(rnn/ourlstm/dense/BiasAdd/ReadVariableOp?'rnn/ourlstm/dense/MatMul/ReadVariableOp?*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_1/MatMul/ReadVariableOp?*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_2/MatMul/ReadVariableOp?*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_3/MatMul/ReadVariableOp?	rnn/whileD
conv2d/Conv2D/ShapeShapex*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
conv2d/Conv2D/ReshapeReshapex$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0`
conv2d_2/Conv2D/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_2/Conv2D/ReshapeReshapeconv2d_1/Relu:activations:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*X
_output_shapesF
D:&??????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
is_training( n
time_distributed/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0m
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????d
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/Reshape_2Reshape(batch_normalization/FusedBatchNormV3:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????0\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_maskb
rnn/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/ourlstm/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros_1:output:0 rnn/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
'rnn/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp0rnn_ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense/MatMulMatMulrnn/ourlstm/concat:output:0/rnn/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(rnn/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp1rnn_ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense/BiasAddBiasAdd"rnn/ourlstm/dense/MatMul:product:00rnn/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
rnn/ourlstm/dense/SigmoidSigmoid"rnn/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
rnn/ourlstm/MulMulrnn/ourlstm/dense/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_1/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_1/BiasAddBiasAdd$rnn/ourlstm/dense_1/MatMul:product:02rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_1/SigmoidSigmoid$rnn/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_2/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_2/BiasAddBiasAdd$rnn/ourlstm/dense_2/MatMul:product:02rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
rnn/ourlstm/dense_2/TanhTanh$rnn/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_1Mulrnn/ourlstm/dense_1/Sigmoid:y:0rnn/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????v
rnn/ourlstm/AddAddV2rnn/ourlstm/Mul:z:0rnn/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_3/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_3/BiasAddBiasAdd$rnn/ourlstm/dense_3/MatMul:product:02rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_3/SigmoidSigmoid$rnn/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
rnn/ourlstm/TanhTanhrnn/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_2Mulrnn/ourlstm/dense_3/Sigmoid:y:0rnn/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:00rnn_ourlstm_dense_matmul_readvariableop_resource1rnn_ourlstm_dense_biasadd_readvariableop_resource2rnn_ourlstm_dense_1_matmul_readvariableop_resource3rnn_ourlstm_dense_1_biasadd_readvariableop_resource2rnn_ourlstm_dense_2_matmul_readvariableop_resource3rnn_ourlstm_dense_2_biasadd_readvariableop_resource2rnn_ourlstm_dense_3_matmul_readvariableop_resource3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_23612* 
condR
rnn_while_cond_23611*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/mul_1Mulrnn/transpose_1:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :???????????????????
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :???????????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_4/Tensordot/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0!dense_4/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????t
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp)^rnn/ourlstm/dense/BiasAdd/ReadVariableOp(^rnn/ourlstm/dense/MatMul/ReadVariableOp+^rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_1/MatMul/ReadVariableOp+^rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_2/MatMul/ReadVariableOp+^rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2T
(rnn/ourlstm/dense/BiasAdd/ReadVariableOp(rnn/ourlstm/dense/BiasAdd/ReadVariableOp2R
'rnn/ourlstm/dense/MatMul/ReadVariableOp'rnn/ourlstm/dense/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_1/MatMul/ReadVariableOp)rnn/ourlstm/dense_1/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_2/MatMul/ReadVariableOp)rnn/ourlstm/dense_2/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_3/MatMul/ReadVariableOp)rnn/ourlstm/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23831

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
is_training( ?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
?[
?
rnn_while_body_25629$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0J
8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0:8G
9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorH
6rnn_while_ourlstm_dense_matmul_readvariableop_resource:8E
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource:??.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp?-rnn/while/ourlstm/dense/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0h
rnn/while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/while/ourlstm/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_3&rnn/while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
-rnn/while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
rnn/while/ourlstm/dense/MatMulMatMul!rnn/while/ourlstm/concat:output:05rnn/while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
rnn/while/ourlstm/dense/BiasAddBiasAdd(rnn/while/ourlstm/dense/MatMul:product:06rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense/SigmoidSigmoid(rnn/while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/MulMul#rnn/while/ourlstm/dense/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_1/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_1/BiasAddBiasAdd*rnn/while/ourlstm/dense_1/MatMul:product:08rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_1/SigmoidSigmoid*rnn/while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_2/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_2/BiasAddBiasAdd*rnn/while/ourlstm/dense_2/MatMul:product:08rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense_2/TanhTanh*rnn/while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_1Mul%rnn/while/ourlstm/dense_1/Sigmoid:y:0"rnn/while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/AddAddV2rnn/while/ourlstm/Mul:z:0rnn/while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_3/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_3/BiasAddBiasAdd*rnn/while/ourlstm/dense_3/MatMul:product:08rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_3/SigmoidSigmoid*rnn/while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
rnn/while/ourlstm/TanhTanhrnn/while/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_2Mul%rnn/while/ourlstm/dense_3/Sigmoid:y:0rnn/while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_4Identityrnn/while/ourlstm/Mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????~
rnn/while/Identity_5Identityrnn/while/ourlstm/Add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:??????????
rnn/while/NoOpNoOp/^rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.^rnn/while/ourlstm/dense/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"x
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0"t
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0"r
6rnn_while_ourlstm_dense_matmul_readvariableop_resource8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2`
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp2^
-rnn/while/ourlstm/dense/MatMul/ReadVariableOp-rnn/while/ourlstm/dense/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
__inference_call_25775
xF
,conv2d_conv2d_conv2d_readvariableop_resource:0G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_1_conv2d_conv2d_readvariableop_resource:00I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_2_conv2d_conv2d_readvariableop_resource:00I
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:09
+batch_normalization_readvariableop_resource:0;
-batch_normalization_readvariableop_1_resource:0J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:0L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:0B
0rnn_ourlstm_dense_matmul_readvariableop_resource:8?
1rnn_ourlstm_dense_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_1_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_1_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_2_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_2_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_3_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_3_biasadd_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:B
4batch_normalization_1_cast_2_readvariableop_resource:B
4batch_normalization_1_cast_3_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?#conv2d/Conv2D/Conv2D/ReadVariableOp?0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_1/Conv2D/Conv2D/ReadVariableOp?2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_2/Conv2D/Conv2D/ReadVariableOp?2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?(rnn/ourlstm/dense/BiasAdd/ReadVariableOp?'rnn/ourlstm/dense/MatMul/ReadVariableOp?*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_1/MatMul/ReadVariableOp?*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_2/MatMul/ReadVariableOp?*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_3/MatMul/ReadVariableOp?	rnn/whileD
conv2d/Conv2D/ShapeShapex*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
conv2d/Conv2D/ReshapeReshapex$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0`
conv2d_2/Conv2D/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_2/Conv2D/ReshapeReshapeconv2d_1/Relu:activations:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*X
_output_shapesF
D:&??????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
is_training( n
time_distributed/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0m
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????d
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/Reshape_2Reshape(batch_normalization/FusedBatchNormV3:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????0\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_maskb
rnn/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/ourlstm/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros_1:output:0 rnn/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
'rnn/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp0rnn_ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense/MatMulMatMulrnn/ourlstm/concat:output:0/rnn/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(rnn/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp1rnn_ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense/BiasAddBiasAdd"rnn/ourlstm/dense/MatMul:product:00rnn/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
rnn/ourlstm/dense/SigmoidSigmoid"rnn/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
rnn/ourlstm/MulMulrnn/ourlstm/dense/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_1/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_1/BiasAddBiasAdd$rnn/ourlstm/dense_1/MatMul:product:02rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_1/SigmoidSigmoid$rnn/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_2/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_2/BiasAddBiasAdd$rnn/ourlstm/dense_2/MatMul:product:02rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
rnn/ourlstm/dense_2/TanhTanh$rnn/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_1Mulrnn/ourlstm/dense_1/Sigmoid:y:0rnn/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????v
rnn/ourlstm/AddAddV2rnn/ourlstm/Mul:z:0rnn/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_3/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_3/BiasAddBiasAdd$rnn/ourlstm/dense_3/MatMul:product:02rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_3/SigmoidSigmoid$rnn/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
rnn/ourlstm/TanhTanhrnn/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_2Mulrnn/ourlstm/dense_3/Sigmoid:y:0rnn/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:00rnn_ourlstm_dense_matmul_readvariableop_resource1rnn_ourlstm_dense_biasadd_readvariableop_resource2rnn_ourlstm_dense_1_matmul_readvariableop_resource3rnn_ourlstm_dense_1_biasadd_readvariableop_resource2rnn_ourlstm_dense_2_matmul_readvariableop_resource3rnn_ourlstm_dense_2_biasadd_readvariableop_resource2rnn_ourlstm_dense_3_matmul_readvariableop_resource3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_25629* 
condR
rnn_while_cond_25628*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/mul_1Mulrnn/transpose_1:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :???????????????????
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :???????????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_4/Tensordot/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0!dense_4/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????t
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp)^rnn/ourlstm/dense/BiasAdd/ReadVariableOp(^rnn/ourlstm/dense/MatMul/ReadVariableOp+^rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_1/MatMul/ReadVariableOp+^rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_2/MatMul/ReadVariableOp+^rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2T
(rnn/ourlstm/dense/BiasAdd/ReadVariableOp(rnn/ourlstm/dense/BiasAdd/ReadVariableOp2R
'rnn/ourlstm/dense/MatMul/ReadVariableOp'rnn/ourlstm/dense/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_1/MatMul/ReadVariableOp)rnn/ourlstm/dense_1/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_2/MatMul/ReadVariableOp)rnn/ourlstm/dense_2/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_3/MatMul/ReadVariableOp)rnn/ourlstm/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?U
?
while_body_27185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?%
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26740

inputs?
%conv2d_conv2d_readvariableop_resource:00@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_25830
input_1!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3:00
	unknown_4:0
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:8

unknown_10:

unknown_11:8

unknown_12:

unknown_13:8

unknown_14:

unknown_15:8

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_23809|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?
?
rnn_while_cond_25628$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_25628___redundant_placeholder0;
7rnn_while_rnn_while_cond_25628___redundant_placeholder1;
7rnn_while_rnn_while_cond_25628___redundant_placeholder2;
7rnn_while_rnn_while_cond_25628___redundant_placeholder3;
7rnn_while_rnn_while_cond_25628___redundant_placeholder4;
7rnn_while_rnn_while_cond_25628___redundant_placeholder5;
7rnn_while_rnn_while_cond_25628___redundant_placeholder6;
7rnn_while_rnn_while_cond_25628___redundant_placeholder7;
7rnn_while_rnn_while_cond_25628___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26784

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
is_training( ?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_24721

inputs>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_24617*
condR
while_cond_24616*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?
?
rnn_while_cond_23611$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_23611___redundant_placeholder0;
7rnn_while_rnn_while_cond_23611___redundant_placeholder1;
7rnn_while_rnn_while_cond_23611___redundant_placeholder2;
7rnn_while_rnn_while_cond_23611___redundant_placeholder3;
7rnn_while_rnn_while_cond_23611___redundant_placeholder4;
7rnn_while_rnn_while_cond_23611___redundant_placeholder5;
7rnn_while_rnn_while_cond_23611___redundant_placeholder6;
7rnn_while_rnn_while_cond_23611___redundant_placeholder7;
7rnn_while_rnn_while_cond_23611___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?%
?
A__inference_conv2d_layer_call_and_return_conditional_losses_24452

inputs?
%conv2d_conv2d_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
?%
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24401

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
#__inference_rnn_layer_call_fn_26878
inputs_0
unknown:8
	unknown_0:
	unknown_1:8
	unknown_2:
	unknown_3:8
	unknown_4:
	unknown_5:8
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_24120|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
L
0__inference_time_distributed_layer_call_fn_26823

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23927m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_23927

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
(global_average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23883\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape1global_average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_25043

inputs>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_24939*
condR
while_cond_24938*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?%
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26698

inputs?
%conv2d_conv2d_readvariableop_resource:00@
2squeeze_batch_dims_biasadd_readvariableop_resource:0
identity??Conv2D/Conv2D/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0z
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0v
IdentityIdentityRelu:activations:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0?
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????0: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?[
?
rnn_while_body_26122$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0J
8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0:8G
9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorH
6rnn_while_ourlstm_dense_matmul_readvariableop_resource:8E
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource:??.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp?-rnn/while/ourlstm/dense/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0h
rnn/while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/while/ourlstm/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_3&rnn/while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
-rnn/while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
rnn/while/ourlstm/dense/MatMulMatMul!rnn/while/ourlstm/concat:output:05rnn/while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
rnn/while/ourlstm/dense/BiasAddBiasAdd(rnn/while/ourlstm/dense/MatMul:product:06rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense/SigmoidSigmoid(rnn/while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/MulMul#rnn/while/ourlstm/dense/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_1/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_1/BiasAddBiasAdd*rnn/while/ourlstm/dense_1/MatMul:product:08rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_1/SigmoidSigmoid*rnn/while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_2/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_2/BiasAddBiasAdd*rnn/while/ourlstm/dense_2/MatMul:product:08rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense_2/TanhTanh*rnn/while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_1Mul%rnn/while/ourlstm/dense_1/Sigmoid:y:0"rnn/while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/AddAddV2rnn/while/ourlstm/Mul:z:0rnn/while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_3/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_3/BiasAddBiasAdd*rnn/while/ourlstm/dense_3/MatMul:product:08rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_3/SigmoidSigmoid*rnn/while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
rnn/while/ourlstm/TanhTanhrnn/while/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_2Mul%rnn/while/ourlstm/dense_3/Sigmoid:y:0rnn/while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_4Identityrnn/while/ourlstm/Mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????~
rnn/while/Identity_5Identityrnn/while/ourlstm/Add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:??????????
rnn/while/NoOpNoOp/^rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.^rnn/while/ourlstm/dense/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"x
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0"t
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0"r
6rnn_while_ourlstm_dense_matmul_readvariableop_resource8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2`
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp2^
-rnn/while/ourlstm/dense/MatMul/ReadVariableOp-rnn/while/ourlstm/dense/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_basic_cnn_lstm_layer_call_fn_25936
x!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3:00
	unknown_4:0
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:8

unknown_10:

unknown_11:8

unknown_12:

unknown_13:8

unknown_14:

unknown_15:8

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25209|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23883

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?U
?
while_body_27359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?[
?
rnn_while_body_23612$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0J
8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0:8G
9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorH
6rnn_while_ourlstm_dense_matmul_readvariableop_resource:8E
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource:??.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp?-rnn/while/ourlstm/dense/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0h
rnn/while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/while/ourlstm/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_3&rnn/while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
-rnn/while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
rnn/while/ourlstm/dense/MatMulMatMul!rnn/while/ourlstm/concat:output:05rnn/while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
rnn/while/ourlstm/dense/BiasAddBiasAdd(rnn/while/ourlstm/dense/MatMul:product:06rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense/SigmoidSigmoid(rnn/while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/MulMul#rnn/while/ourlstm/dense/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_1/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_1/BiasAddBiasAdd*rnn/while/ourlstm/dense_1/MatMul:product:08rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_1/SigmoidSigmoid*rnn/while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_2/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_2/BiasAddBiasAdd*rnn/while/ourlstm/dense_2/MatMul:product:08rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense_2/TanhTanh*rnn/while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_1Mul%rnn/while/ourlstm/dense_1/Sigmoid:y:0"rnn/while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/AddAddV2rnn/while/ourlstm/Mul:z:0rnn/while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_3/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_3/BiasAddBiasAdd*rnn/while/ourlstm/dense_3/MatMul:product:08rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_3/SigmoidSigmoid*rnn/while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
rnn/while/ourlstm/TanhTanhrnn/while/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_2Mul%rnn/while/ourlstm/dense_3/Sigmoid:y:0rnn/while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_4Identityrnn/while/ourlstm/Mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????~
rnn/while/Identity_5Identityrnn/while/ourlstm/Add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:??????????
rnn/while/NoOpNoOp/^rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.^rnn/while/ourlstm/dense/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"x
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0"t
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0"r
6rnn_while_ourlstm_dense_matmul_readvariableop_resource8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2`
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp2^
-rnn/while/ourlstm/dense/MatMul/ReadVariableOp-rnn/while/ourlstm/dense/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_basic_cnn_lstm_layer_call_fn_25883
x!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3:00
	unknown_4:0
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:8

unknown_10:

unknown_11:8

unknown_12:

unknown_13:8

unknown_14:

unknown_15:8

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_24785|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?U
?
while_body_24617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?U
?
while_body_27011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_ourlstm_dense_matmul_readvariableop_resource_0:8C
5while_ourlstm_dense_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_1_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_1_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_2_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_2_biasadd_readvariableop_resource_0:H
6while_ourlstm_dense_3_matmul_readvariableop_resource_0:8E
7while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_ourlstm_dense_matmul_readvariableop_resource:8A
3while_ourlstm_dense_biasadd_readvariableop_resource:F
4while_ourlstm_dense_1_matmul_readvariableop_resource:8C
5while_ourlstm_dense_1_biasadd_readvariableop_resource:F
4while_ourlstm_dense_2_matmul_readvariableop_resource:8C
5while_ourlstm_dense_2_biasadd_readvariableop_resource:F
4while_ourlstm_dense_3_matmul_readvariableop_resource:8C
5while_ourlstm_dense_3_biasadd_readvariableop_resource:??*while/ourlstm/dense/BiasAdd/ReadVariableOp?)while/ourlstm/dense/MatMul/ReadVariableOp?,while/ourlstm/dense_1/BiasAdd/ReadVariableOp?+while/ourlstm/dense_1/MatMul/ReadVariableOp?,while/ourlstm/dense_2/BiasAdd/ReadVariableOp?+while/ourlstm/dense_2/MatMul/ReadVariableOp?,while/ourlstm/dense_3/BiasAdd/ReadVariableOp?+while/ourlstm/dense_3/MatMul/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0d
while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/ourlstm/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_3"while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
)while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp4while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense/MatMulMatMulwhile/ourlstm/concat:output:01while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp5while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense/BiasAddBiasAdd$while/ourlstm/dense/MatMul:product:02while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
while/ourlstm/dense/SigmoidSigmoid$while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/MulMulwhile/ourlstm/dense/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_1/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_1/BiasAddBiasAdd&while/ourlstm/dense_1/MatMul:product:04while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_1/SigmoidSigmoid&while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_2/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_2/BiasAddBiasAdd&while/ourlstm/dense_2/MatMul:product:04while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/dense_2/TanhTanh&while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_1Mul!while/ourlstm/dense_1/Sigmoid:y:0while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????|
while/ourlstm/AddAddV2while/ourlstm/Mul:z:0while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
+while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp6while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
while/ourlstm/dense_3/MatMulMatMulwhile/ourlstm/concat:output:03while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp7while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/ourlstm/dense_3/BiasAddBiasAdd&while/ourlstm/dense_3/MatMul:product:04while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/ourlstm/dense_3/SigmoidSigmoid&while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
while/ourlstm/TanhTanhwhile/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
while/ourlstm/Mul_2Mul!while/ourlstm/dense_3/Sigmoid:y:0while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/ourlstm/Mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????r
while/Identity_5Identitywhile/ourlstm/Add:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp+^while/ourlstm/dense/BiasAdd/ReadVariableOp*^while/ourlstm/dense/MatMul/ReadVariableOp-^while/ourlstm/dense_1/BiasAdd/ReadVariableOp,^while/ourlstm/dense_1/MatMul/ReadVariableOp-^while/ourlstm/dense_2/BiasAdd/ReadVariableOp,^while/ourlstm/dense_2/MatMul/ReadVariableOp-^while/ourlstm/dense_3/BiasAdd/ReadVariableOp,^while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"p
5while_ourlstm_dense_1_biasadd_readvariableop_resource7while_ourlstm_dense_1_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_1_matmul_readvariableop_resource6while_ourlstm_dense_1_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_2_biasadd_readvariableop_resource7while_ourlstm_dense_2_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_2_matmul_readvariableop_resource6while_ourlstm_dense_2_matmul_readvariableop_resource_0"p
5while_ourlstm_dense_3_biasadd_readvariableop_resource7while_ourlstm_dense_3_biasadd_readvariableop_resource_0"n
4while_ourlstm_dense_3_matmul_readvariableop_resource6while_ourlstm_dense_3_matmul_readvariableop_resource_0"l
3while_ourlstm_dense_biasadd_readvariableop_resource5while_ourlstm_dense_biasadd_readvariableop_resource_0"j
2while_ourlstm_dense_matmul_readvariableop_resource4while_ourlstm_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/ourlstm/dense/BiasAdd/ReadVariableOp*while/ourlstm/dense/BiasAdd/ReadVariableOp2V
)while/ourlstm/dense/MatMul/ReadVariableOp)while/ourlstm/dense/MatMul/ReadVariableOp2\
,while/ourlstm/dense_1/BiasAdd/ReadVariableOp,while/ourlstm/dense_1/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_1/MatMul/ReadVariableOp+while/ourlstm/dense_1/MatMul/ReadVariableOp2\
,while/ourlstm/dense_2/BiasAdd/ReadVariableOp,while/ourlstm/dense_2/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_2/MatMul/ReadVariableOp+while/ourlstm/dense_2/MatMul/ReadVariableOp2\
,while/ourlstm/dense_3/BiasAdd/ReadVariableOp,while/ourlstm/dense_3/BiasAdd/ReadVariableOp2Z
+while/ourlstm/dense_3/MatMul/ReadVariableOp+while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_27289
inputs_0>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_27185*
condR
while_cond_27184*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_26840

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
global_average_pooling2d/MeanMeanReshape:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape&global_average_pooling2d/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
L
0__inference_time_distributed_layer_call_fn_26818

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23906m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23862

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_26623

inputs!
unknown:0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_24452?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&??????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27683

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
while_cond_24938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24938___redundant_placeholder03
/while_while_cond_24938___redundant_placeholder13
/while_while_cond_24938___redundant_placeholder23
/while_while_cond_24938___redundant_placeholder33
/while_while_cond_24938___redundant_placeholder43
/while_while_cond_24938___redundant_placeholder53
/while_while_cond_24938___redundant_placeholder63
/while_while_cond_24938___redundant_placeholder73
/while_while_cond_24938___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_basic_cnn_lstm_layer_call_fn_24836
input_1!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3:00
	unknown_4:0
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:8

unknown_10:

unknown_11:8

unknown_12:

unknown_13:8

unknown_14:

unknown_15:8

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_24785|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?'
?	
while_body_24032
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0'
while_ourlstm_24056_0:8#
while_ourlstm_24058_0:'
while_ourlstm_24060_0:8#
while_ourlstm_24062_0:'
while_ourlstm_24064_0:8#
while_ourlstm_24066_0:'
while_ourlstm_24068_0:8#
while_ourlstm_24070_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor%
while_ourlstm_24056:8!
while_ourlstm_24058:%
while_ourlstm_24060:8!
while_ourlstm_24062:%
while_ourlstm_24064:8!
while_ourlstm_24066:%
while_ourlstm_24068:8!
while_ourlstm_24070:??%while/ourlstm/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0?
%while/ourlstm/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ourlstm_24056_0while_ourlstm_24058_0while_ourlstm_24060_0while_ourlstm_24062_0while_ourlstm_24064_0while_ourlstm_24066_0while_ourlstm_24068_0while_ourlstm_24070_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder.while/ourlstm/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity.while/ourlstm/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity.while/ourlstm/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????t

while/NoOpNoOp&^while/ourlstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_ourlstm_24056while_ourlstm_24056_0",
while_ourlstm_24058while_ourlstm_24058_0",
while_ourlstm_24060while_ourlstm_24060_0",
while_ourlstm_24062while_ourlstm_24062_0",
while_ourlstm_24064while_ourlstm_24064_0",
while_ourlstm_24066while_ourlstm_24066_0",
while_ourlstm_24068while_ourlstm_24068_0",
while_ourlstm_24070while_ourlstm_24070_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2N
%while/ourlstm/StatefulPartitionedCall%while/ourlstm/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

?
while_cond_27358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_27358___redundant_placeholder03
/while_while_cond_27358___redundant_placeholder13
/while_while_cond_27358___redundant_placeholder23
/while_while_cond_27358___redundant_placeholder33
/while_while_cond_27358___redundant_placeholder43
/while_while_cond_27358___redundant_placeholder53
/while_while_cond_27358___redundant_placeholder63
/while_while_cond_27358___redundant_placeholder73
/while_while_cond_27358___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?1
?	
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25375
input_1&
conv2d_25316:0
conv2d_25318:0(
conv2d_1_25321:00
conv2d_1_25323:0(
conv2d_2_25326:00
conv2d_2_25328:0'
batch_normalization_25331:0'
batch_normalization_25333:0'
batch_normalization_25335:0'
batch_normalization_25337:0
	rnn_25343:8
	rnn_25345:
	rnn_25347:8
	rnn_25349:
	rnn_25351:8
	rnn_25353:
	rnn_25355:8
	rnn_25357:)
batch_normalization_1_25360:)
batch_normalization_1_25362:)
batch_normalization_1_25364:)
batch_normalization_1_25366:
dense_4_25369:
dense_4_25371:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_25316conv2d_25318*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_24452?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_25321conv2d_1_25323*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24491?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_25326conv2d_2_25328*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24530?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_25331batch_normalization_25333batch_normalization_25335batch_normalization_25337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23831?
 time_distributed/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_23906w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape4batch_normalization/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_25343	rnn_25345	rnn_25347	rnn_25349	rnn_25351	rnn_25353	rnn_25355	rnn_25357*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_24721?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0batch_normalization_1_25360batch_normalization_1_25362batch_normalization_1_25364batch_normalization_1_25366*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24354?
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_4_25369dense_4_25371*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_24778?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?'
?	
while_body_24223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0'
while_ourlstm_24247_0:8#
while_ourlstm_24249_0:'
while_ourlstm_24251_0:8#
while_ourlstm_24253_0:'
while_ourlstm_24255_0:8#
while_ourlstm_24257_0:'
while_ourlstm_24259_0:8#
while_ourlstm_24261_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor%
while_ourlstm_24247:8!
while_ourlstm_24249:%
while_ourlstm_24251:8!
while_ourlstm_24253:%
while_ourlstm_24255:8!
while_ourlstm_24257:%
while_ourlstm_24259:8!
while_ourlstm_24261:??%while/ourlstm/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0?
%while/ourlstm/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ourlstm_24247_0while_ourlstm_24249_0while_ourlstm_24251_0while_ourlstm_24253_0while_ourlstm_24255_0while_ourlstm_24257_0while_ourlstm_24259_0while_ourlstm_24261_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_ourlstm_layer_call_and_return_conditional_losses_24008?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder.while/ourlstm/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity.while/ourlstm/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity.while/ourlstm/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????t

while/NoOpNoOp&^while/ourlstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_ourlstm_24247while_ourlstm_24247_0",
while_ourlstm_24249while_ourlstm_24249_0",
while_ourlstm_24251while_ourlstm_24251_0",
while_ourlstm_24253while_ourlstm_24253_0",
while_ourlstm_24255while_ourlstm_24255_0",
while_ourlstm_24257while_ourlstm_24257_0",
while_ourlstm_24259while_ourlstm_24259_0",
while_ourlstm_24261while_ourlstm_24261_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2N
%while/ourlstm/StatefulPartitionedCall%while/ourlstm/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
 __inference__wrapped_model_23809
input_1.
basic_cnn_lstm_23759:0"
basic_cnn_lstm_23761:0.
basic_cnn_lstm_23763:00"
basic_cnn_lstm_23765:0.
basic_cnn_lstm_23767:00"
basic_cnn_lstm_23769:0"
basic_cnn_lstm_23771:0"
basic_cnn_lstm_23773:0"
basic_cnn_lstm_23775:0"
basic_cnn_lstm_23777:0&
basic_cnn_lstm_23779:8"
basic_cnn_lstm_23781:&
basic_cnn_lstm_23783:8"
basic_cnn_lstm_23785:&
basic_cnn_lstm_23787:8"
basic_cnn_lstm_23789:&
basic_cnn_lstm_23791:8"
basic_cnn_lstm_23793:"
basic_cnn_lstm_23795:"
basic_cnn_lstm_23797:"
basic_cnn_lstm_23799:"
basic_cnn_lstm_23801:&
basic_cnn_lstm_23803:"
basic_cnn_lstm_23805:
identity??&basic_cnn_lstm/StatefulPartitionedCall?
&basic_cnn_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1basic_cnn_lstm_23759basic_cnn_lstm_23761basic_cnn_lstm_23763basic_cnn_lstm_23765basic_cnn_lstm_23767basic_cnn_lstm_23769basic_cnn_lstm_23771basic_cnn_lstm_23773basic_cnn_lstm_23775basic_cnn_lstm_23777basic_cnn_lstm_23779basic_cnn_lstm_23781basic_cnn_lstm_23783basic_cnn_lstm_23785basic_cnn_lstm_23787basic_cnn_lstm_23789basic_cnn_lstm_23791basic_cnn_lstm_23793basic_cnn_lstm_23795basic_cnn_lstm_23797basic_cnn_lstm_23799basic_cnn_lstm_23801basic_cnn_lstm_23803basic_cnn_lstm_23805*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_23758?
IdentityIdentity/basic_cnn_lstm/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????o
NoOpNoOp'^basic_cnn_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2P
&basic_cnn_lstm/StatefulPartitionedCall&basic_cnn_lstm/StatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_27115
inputs_0>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_27011*
condR
while_cond_27010*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_23906

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
(global_average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23883\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape1global_average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????0:d `
<
_output_shapes*
(:&??????????????????0
 
_user_specified_nameinputs
?[
?
rnn_while_body_26454$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0J
8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0:8G
9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0:L
:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0:8I
;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorH
6rnn_while_ourlstm_dense_matmul_readvariableop_resource:8E
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource:J
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:8G
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource:??.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp?-rnn/while/ourlstm/dense/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp?0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp?/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????0*
element_dtype0h
rnn/while/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/while/ourlstm/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_3&rnn/while/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
-rnn/while/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
rnn/while/ourlstm/dense/MatMulMatMul!rnn/while/ourlstm/concat:output:05rnn/while/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
rnn/while/ourlstm/dense/BiasAddBiasAdd(rnn/while/ourlstm/dense/MatMul:product:06rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense/SigmoidSigmoid(rnn/while/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/MulMul#rnn/while/ourlstm/dense/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_1/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_1/BiasAddBiasAdd*rnn/while/ourlstm/dense_1/MatMul:product:08rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_1/SigmoidSigmoid*rnn/while/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_2/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_2/BiasAddBiasAdd*rnn/while/ourlstm/dense_2/MatMul:product:08rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/dense_2/TanhTanh*rnn/while/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_1Mul%rnn/while/ourlstm/dense_1/Sigmoid:y:0"rnn/while/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/AddAddV2rnn/while/ourlstm/Mul:z:0rnn/while/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:8*
dtype0?
 rnn/while/ourlstm/dense_3/MatMulMatMul!rnn/while/ourlstm/concat:output:07rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
!rnn/while/ourlstm/dense_3/BiasAddBiasAdd*rnn/while/ourlstm/dense_3/MatMul:product:08rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!rnn/while/ourlstm/dense_3/SigmoidSigmoid*rnn/while/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
rnn/while/ourlstm/TanhTanhrnn/while/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/while/ourlstm/Mul_2Mul%rnn/while/ourlstm/dense_3/Sigmoid:y:0rnn/while/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/ourlstm/Mul_2:z:0*
_output_shapes
: *
element_dtype0:???Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: ?
rnn/while/Identity_4Identityrnn/while/ourlstm/Mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????~
rnn/while/Identity_5Identityrnn/while/ourlstm/Add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:??????????
rnn/while/NoOpNoOp/^rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.^rnn/while/ourlstm/dense/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp1^rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0^rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"x
9rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_1_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_1_matmul_readvariableop_resource:rnn_while_ourlstm_dense_1_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_2_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_2_matmul_readvariableop_resource:rnn_while_ourlstm_dense_2_matmul_readvariableop_resource_0"x
9rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource;rnn_while_ourlstm_dense_3_biasadd_readvariableop_resource_0"v
8rnn_while_ourlstm_dense_3_matmul_readvariableop_resource:rnn_while_ourlstm_dense_3_matmul_readvariableop_resource_0"t
7rnn_while_ourlstm_dense_biasadd_readvariableop_resource9rnn_while_ourlstm_dense_biasadd_readvariableop_resource_0"r
6rnn_while_ourlstm_dense_matmul_readvariableop_resource8rnn_while_ourlstm_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2`
.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp.rnn/while/ourlstm/dense/BiasAdd/ReadVariableOp2^
-rnn/while/ourlstm/dense/MatMul/ReadVariableOp-rnn/while/ourlstm/dense/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_1/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_1/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_2/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_2/MatMul/ReadVariableOp2d
0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp0rnn/while/ourlstm/dense_3/BiasAdd/ReadVariableOp2b
/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp/rnn/while/ourlstm/dense_3/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_basic_cnn_lstm_layer_call_fn_25313
input_1!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0#
	unknown_3:00
	unknown_4:0
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:8

unknown_10:

unknown_11:8

unknown_12:

unknown_13:8

unknown_14:

unknown_15:8

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25209|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
<
_output_shapes*
(:&??????????????????
!
_user_specified_name	input_1
?,
?
B__inference_ourlstm_layer_call_and_return_conditional_losses_27826

inputs
states_0
states_16
$dense_matmul_readvariableop_resource:83
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:85
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:85
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:85
'dense_3_biasadd_readvariableop_resource:
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????u
concatConcatV2inputsstates_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0~
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Y
MulMuldense/Sigmoid:y:0states_0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_1/MatMulMatMulconcat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_2/MatMulMatMulconcat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
Mul_1Muldense_1/Sigmoid:y:0dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????R
AddAddV2Mul:z:0	Mul_1:z:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????G
TanhTanhAdd:z:0*
T0*'
_output_shapes
:?????????]
Mul_2Muldense_3/Sigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	Mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	Mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????X

Identity_2IdentityAdd:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????0:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
ް
?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26268
xF
,conv2d_conv2d_conv2d_readvariableop_resource:0G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_1_conv2d_conv2d_readvariableop_resource:00I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:0H
.conv2d_2_conv2d_conv2d_readvariableop_resource:00I
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:09
+batch_normalization_readvariableop_resource:0;
-batch_normalization_readvariableop_1_resource:0J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:0L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:0B
0rnn_ourlstm_dense_matmul_readvariableop_resource:8?
1rnn_ourlstm_dense_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_1_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_1_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_2_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_2_biasadd_readvariableop_resource:D
2rnn_ourlstm_dense_3_matmul_readvariableop_resource:8A
3rnn_ourlstm_dense_3_biasadd_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:B
4batch_normalization_1_cast_2_readvariableop_resource:B
4batch_normalization_1_cast_3_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?#conv2d/Conv2D/Conv2D/ReadVariableOp?0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_1/Conv2D/Conv2D/ReadVariableOp?2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp?%conv2d_2/Conv2D/Conv2D/ReadVariableOp?2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?(rnn/ourlstm/dense/BiasAdd/ReadVariableOp?'rnn/ourlstm/dense/MatMul/ReadVariableOp?*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_1/MatMul/ReadVariableOp?*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_2/MatMul/ReadVariableOp?*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp?)rnn/ourlstm/dense_3/MatMul/ReadVariableOp?	rnn/whileD
conv2d/Conv2D/ShapeShapex*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
conv2d/Conv2D/ReshapeReshapex$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0^
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0`
conv2d_2/Conv2D/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
conv2d_2/Conv2D/ReshapeReshapeconv2d_1/Relu:activations:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0s
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      0   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????0?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*X
_output_shapesF
D:&??????????????????0:0:0:0:0:*
data_formatNDHWC*
epsilon%o?:*
is_training( n
time_distributed/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0?
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????0m
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????d
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :0?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????0y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   ?
time_distributed/Reshape_2Reshape(batch_normalization/FusedBatchNormV3:y:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????0\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_maskb
rnn/ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn/ourlstm/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros_1:output:0 rnn/ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
'rnn/ourlstm/dense/MatMul/ReadVariableOpReadVariableOp0rnn_ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense/MatMulMatMulrnn/ourlstm/concat:output:0/rnn/ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(rnn/ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp1rnn_ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense/BiasAddBiasAdd"rnn/ourlstm/dense/MatMul:product:00rnn/ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
rnn/ourlstm/dense/SigmoidSigmoid"rnn/ourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
rnn/ourlstm/MulMulrnn/ourlstm/dense/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_1/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_1/BiasAddBiasAdd$rnn/ourlstm/dense_1/MatMul:product:02rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_1/SigmoidSigmoid$rnn/ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_2/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_2/BiasAddBiasAdd$rnn/ourlstm/dense_2/MatMul:product:02rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
rnn/ourlstm/dense_2/TanhTanh$rnn/ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_1Mulrnn/ourlstm/dense_1/Sigmoid:y:0rnn/ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????v
rnn/ourlstm/AddAddV2rnn/ourlstm/Mul:z:0rnn/ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)rnn/ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp2rnn_ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
rnn/ourlstm/dense_3/MatMulMatMulrnn/ourlstm/concat:output:01rnn/ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
rnn/ourlstm/dense_3/BiasAddBiasAdd$rnn/ourlstm/dense_3/MatMul:product:02rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
rnn/ourlstm/dense_3/SigmoidSigmoid$rnn/ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
rnn/ourlstm/TanhTanhrnn/ourlstm/Add:z:0*
T0*'
_output_shapes
:??????????
rnn/ourlstm/Mul_2Mulrnn/ourlstm/dense_3/Sigmoid:y:0rnn/ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:00rnn_ourlstm_dense_matmul_readvariableop_resource1rnn_ourlstm_dense_biasadd_readvariableop_resource2rnn_ourlstm_dense_1_matmul_readvariableop_resource3rnn_ourlstm_dense_1_biasadd_readvariableop_resource2rnn_ourlstm_dense_2_matmul_readvariableop_resource3rnn_ourlstm_dense_2_biasadd_readvariableop_resource2rnn_ourlstm_dense_3_matmul_readvariableop_resource3rnn_ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_26122* 
condR
rnn_while_cond_26121*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/mul_1Mulrnn/transpose_1:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :???????????????????
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :???????????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_4/Tensordot/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0!dense_4/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????t
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp)^rnn/ourlstm/dense/BiasAdd/ReadVariableOp(^rnn/ourlstm/dense/MatMul/ReadVariableOp+^rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_1/MatMul/ReadVariableOp+^rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_2/MatMul/ReadVariableOp+^rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*^rnn/ourlstm/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:&??????????????????: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2T
(rnn/ourlstm/dense/BiasAdd/ReadVariableOp(rnn/ourlstm/dense/BiasAdd/ReadVariableOp2R
'rnn/ourlstm/dense/MatMul/ReadVariableOp'rnn/ourlstm/dense/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_1/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_1/MatMul/ReadVariableOp)rnn/ourlstm/dense_1/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_2/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_2/MatMul/ReadVariableOp)rnn/ourlstm/dense_2/MatMul/ReadVariableOp2X
*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp*rnn/ourlstm/dense_3/BiasAdd/ReadVariableOp2V
)rnn/ourlstm/dense_3/MatMul/ReadVariableOp)rnn/ourlstm/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:_ [
<
_output_shapes*
(:&??????????????????

_user_specified_namex
?_
?
>__inference_rnn_layer_call_and_return_conditional_losses_27463

inputs>
,ourlstm_dense_matmul_readvariableop_resource:8;
-ourlstm_dense_biasadd_readvariableop_resource:@
.ourlstm_dense_1_matmul_readvariableop_resource:8=
/ourlstm_dense_1_biasadd_readvariableop_resource:@
.ourlstm_dense_2_matmul_readvariableop_resource:8=
/ourlstm_dense_2_biasadd_readvariableop_resource:@
.ourlstm_dense_3_matmul_readvariableop_resource:8=
/ourlstm_dense_3_biasadd_readvariableop_resource:
identity??$ourlstm/dense/BiasAdd/ReadVariableOp?#ourlstm/dense/MatMul/ReadVariableOp?&ourlstm/dense_1/BiasAdd/ReadVariableOp?%ourlstm/dense_1/MatMul/ReadVariableOp?&ourlstm/dense_2/BiasAdd/ReadVariableOp?%ourlstm/dense_2/MatMul/ReadVariableOp?&ourlstm/dense_3/BiasAdd/ReadVariableOp?%ourlstm/dense_3/MatMul/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????0   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????0*
shrink_axis_mask^
ourlstm/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ourlstm/concatConcatV2strided_slice_2:output:0zeros_1:output:0ourlstm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????8?
#ourlstm/dense/MatMul/ReadVariableOpReadVariableOp,ourlstm_dense_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense/MatMulMatMulourlstm/concat:output:0+ourlstm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$ourlstm/dense/BiasAdd/ReadVariableOpReadVariableOp-ourlstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense/BiasAddBiasAddourlstm/dense/MatMul:product:0,ourlstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
ourlstm/dense/SigmoidSigmoidourlstm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
ourlstm/MulMulourlstm/dense/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_1/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_1_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_1/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_1/BiasAddBiasAdd ourlstm/dense_1/MatMul:product:0.ourlstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_1/SigmoidSigmoid ourlstm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_2/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_2_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_2/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_2/BiasAddBiasAdd ourlstm/dense_2/MatMul:product:0.ourlstm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
ourlstm/dense_2/TanhTanh ourlstm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
ourlstm/Mul_1Mulourlstm/dense_1/Sigmoid:y:0ourlstm/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
ourlstm/AddAddV2ourlstm/Mul:z:0ourlstm/Mul_1:z:0*
T0*'
_output_shapes
:??????????
%ourlstm/dense_3/MatMul/ReadVariableOpReadVariableOp.ourlstm_dense_3_matmul_readvariableop_resource*
_output_shapes

:8*
dtype0?
ourlstm/dense_3/MatMulMatMulourlstm/concat:output:0-ourlstm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&ourlstm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/ourlstm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ourlstm/dense_3/BiasAddBiasAdd ourlstm/dense_3/MatMul:product:0.ourlstm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
ourlstm/dense_3/SigmoidSigmoid ourlstm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
ourlstm/TanhTanhourlstm/Add:z:0*
T0*'
_output_shapes
:?????????u
ourlstm/Mul_2Mulourlstm/dense_3/Sigmoid:y:0ourlstm/Tanh:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,ourlstm_dense_matmul_readvariableop_resource-ourlstm_dense_biasadd_readvariableop_resource.ourlstm_dense_1_matmul_readvariableop_resource/ourlstm_dense_1_biasadd_readvariableop_resource.ourlstm_dense_2_matmul_readvariableop_resource/ourlstm_dense_2_biasadd_readvariableop_resource.ourlstm_dense_3_matmul_readvariableop_resource/ourlstm_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_27359*
condR
while_cond_27358*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp%^ourlstm/dense/BiasAdd/ReadVariableOp$^ourlstm/dense/MatMul/ReadVariableOp'^ourlstm/dense_1/BiasAdd/ReadVariableOp&^ourlstm/dense_1/MatMul/ReadVariableOp'^ourlstm/dense_2/BiasAdd/ReadVariableOp&^ourlstm/dense_2/MatMul/ReadVariableOp'^ourlstm/dense_3/BiasAdd/ReadVariableOp&^ourlstm/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 2L
$ourlstm/dense/BiasAdd/ReadVariableOp$ourlstm/dense/BiasAdd/ReadVariableOp2J
#ourlstm/dense/MatMul/ReadVariableOp#ourlstm/dense/MatMul/ReadVariableOp2P
&ourlstm/dense_1/BiasAdd/ReadVariableOp&ourlstm/dense_1/BiasAdd/ReadVariableOp2N
%ourlstm/dense_1/MatMul/ReadVariableOp%ourlstm/dense_1/MatMul/ReadVariableOp2P
&ourlstm/dense_2/BiasAdd/ReadVariableOp&ourlstm/dense_2/BiasAdd/ReadVariableOp2N
%ourlstm/dense_2/MatMul/ReadVariableOp%ourlstm/dense_2/MatMul/ReadVariableOp2P
&ourlstm/dense_3/BiasAdd/ReadVariableOp&ourlstm/dense_3/BiasAdd/ReadVariableOp2N
%ourlstm/dense_3/MatMul/ReadVariableOp%ourlstm/dense_3/MatMul/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_26766

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23862?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????0
 
_user_specified_nameinputs
?%
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27717

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
#__inference_rnn_layer_call_fn_26920

inputs
unknown:8
	unknown_0:
	unknown_1:8
	unknown_2:
	unknown_3:8
	unknown_4:
	unknown_5:8
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_24721|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????0: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
??
?!
__inference__traced_save_28068
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop7
3savev2_rnn_ourlstm_dense_kernel_read_readvariableop5
1savev2_rnn_ourlstm_dense_bias_read_readvariableop9
5savev2_rnn_ourlstm_dense_1_kernel_read_readvariableop7
3savev2_rnn_ourlstm_dense_1_bias_read_readvariableop9
5savev2_rnn_ourlstm_dense_2_kernel_read_readvariableop7
3savev2_rnn_ourlstm_dense_2_bias_read_readvariableop9
5savev2_rnn_ourlstm_dense_3_kernel_read_readvariableop7
3savev2_rnn_ourlstm_dense_3_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_kernel_m_read_readvariableop<
8savev2_adam_rnn_ourlstm_dense_bias_m_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_1_kernel_m_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_1_bias_m_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_2_kernel_m_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_2_bias_m_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_3_kernel_m_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_kernel_v_read_readvariableop<
8savev2_adam_rnn_ourlstm_dense_bias_v_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_1_kernel_v_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_1_bias_v_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_2_kernel_v_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_2_bias_v_read_readvariableop@
<savev2_adam_rnn_ourlstm_dense_3_kernel_v_read_readvariableop>
:savev2_adam_rnn_ourlstm_dense_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*? 
value? B? JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop3savev2_rnn_ourlstm_dense_kernel_read_readvariableop1savev2_rnn_ourlstm_dense_bias_read_readvariableop5savev2_rnn_ourlstm_dense_1_kernel_read_readvariableop3savev2_rnn_ourlstm_dense_1_bias_read_readvariableop5savev2_rnn_ourlstm_dense_2_kernel_read_readvariableop3savev2_rnn_ourlstm_dense_2_bias_read_readvariableop5savev2_rnn_ourlstm_dense_3_kernel_read_readvariableop3savev2_rnn_ourlstm_dense_3_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop:savev2_adam_rnn_ourlstm_dense_kernel_m_read_readvariableop8savev2_adam_rnn_ourlstm_dense_bias_m_read_readvariableop<savev2_adam_rnn_ourlstm_dense_1_kernel_m_read_readvariableop:savev2_adam_rnn_ourlstm_dense_1_bias_m_read_readvariableop<savev2_adam_rnn_ourlstm_dense_2_kernel_m_read_readvariableop:savev2_adam_rnn_ourlstm_dense_2_bias_m_read_readvariableop<savev2_adam_rnn_ourlstm_dense_3_kernel_m_read_readvariableop:savev2_adam_rnn_ourlstm_dense_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop:savev2_adam_rnn_ourlstm_dense_kernel_v_read_readvariableop8savev2_adam_rnn_ourlstm_dense_bias_v_read_readvariableop<savev2_adam_rnn_ourlstm_dense_1_kernel_v_read_readvariableop:savev2_adam_rnn_ourlstm_dense_1_bias_v_read_readvariableop<savev2_adam_rnn_ourlstm_dense_2_kernel_v_read_readvariableop:savev2_adam_rnn_ourlstm_dense_2_bias_v_read_readvariableop<savev2_adam_rnn_ourlstm_dense_3_kernel_v_read_readvariableop:savev2_adam_rnn_ourlstm_dense_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :0:0:00:0:00:0:0:0:0:0:8::8::8::8:::::::: : : : : : : : : :0:0:00:0:00:0:0:0:8::8::8::8::::::0:0:00:0:00:0:0:0:8::8::8::8:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 	

_output_shapes
:0: 


_output_shapes
:0:$ 

_output_shapes

:8: 

_output_shapes
::$ 

_output_shapes

:8: 

_output_shapes
::$ 

_output_shapes

:8: 

_output_shapes
::$ 

_output_shapes

:8: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
:0: #

_output_shapes
:0:,$(
&
_output_shapes
:00: %

_output_shapes
:0:,&(
&
_output_shapes
:00: '

_output_shapes
:0: (

_output_shapes
:0: )

_output_shapes
:0:$* 

_output_shapes

:8: +

_output_shapes
::$, 

_output_shapes

:8: -

_output_shapes
::$. 

_output_shapes

:8: /

_output_shapes
::$0 

_output_shapes

:8: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::,6(
&
_output_shapes
:0: 7

_output_shapes
:0:,8(
&
_output_shapes
:00: 9

_output_shapes
:0:,:(
&
_output_shapes
:00: ;

_output_shapes
:0: <

_output_shapes
:0: =

_output_shapes
:0:$> 

_output_shapes

:8: ?

_output_shapes
::$@ 

_output_shapes

:8: A

_output_shapes
::$B 

_output_shapes

:8: C

_output_shapes
::$D 

_output_shapes

:8: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::J

_output_shapes
: "?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
P
input_1E
serving_default_input_1:0&??????????????????I
output_1=
StatefulPartitionedCall:0??????????????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

convlayer1
	
convlayer2


convlayer3

batchnorm1
global_pool
timedist
rnn

batchnorm2
outputlayer
	optimizer
call

signatures"
_tf_keras_model
?
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
*18
+19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
1trace_0
2trace_1
3trace_2
4trace_32?
.__inference_basic_cnn_lstm_layer_call_fn_24836
.__inference_basic_cnn_lstm_layer_call_fn_25883
.__inference_basic_cnn_lstm_layer_call_fn_25936
.__inference_basic_cnn_lstm_layer_call_fn_25313?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z1trace_0z2trace_1z3trace_2z4trace_3
?
5trace_0
6trace_1
7trace_2
8trace_32?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26268
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26614
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25375
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25437?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z5trace_0z6trace_1z7trace_2z8trace_3
?B?
 __inference__wrapped_model_23809input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias
 ?_jit_compiled_convolution_op"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

kernel
bias
 F_jit_compiled_convolution_op"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op"
_tf_keras_layer
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
	layer"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
gcell
h
state_spec"
_tf_keras_rnn_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oaxis
	&gamma
'beta
(moving_mean
)moving_variance"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
?
viter

wbeta_1

xbeta_2
	ydecay
zlearning_ratem?m?m?m?m?m?m?m?m?m? m?!m?"m?#m?$m?%m?&m?'m?*m?+m?v?v?v?v?v?v?v?v?v?v? v?!v?"v?#v?$v?%v?&v?'v?*v?+v?"
	optimizer
?
{trace_02?
__inference_call_25775?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z{trace_0
,
|serving_default"
signature_map
':%02conv2d/kernel
:02conv2d/bias
):'002conv2d_1/kernel
:02conv2d_1/bias
):'002conv2d_2/kernel
:02conv2d_2/bias
':%02batch_normalization/gamma
&:$02batch_normalization/beta
/:-0 (2batch_normalization/moving_mean
3:10 (2#batch_normalization/moving_variance
*:(82rnn/ourlstm/dense/kernel
$:"2rnn/ourlstm/dense/bias
,:*82rnn/ourlstm/dense_1/kernel
&:$2rnn/ourlstm/dense_1/bias
,:*82rnn/ourlstm/dense_2/kernel
&:$2rnn/ourlstm/dense_2/bias
,:*82rnn/ourlstm/dense_3/kernel
&:$2rnn/ourlstm/dense_3/bias
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
 :2dense_4/kernel
:2dense_4/bias
<
0
1
(2
)3"
trackable_list_wrapper
_
0
	1

2
3
4
5
6
7
8"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_basic_cnn_lstm_layer_call_fn_24836input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
.__inference_basic_cnn_lstm_layer_call_fn_25883x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
.__inference_basic_cnn_lstm_layer_call_fn_25936x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
.__inference_basic_cnn_lstm_layer_call_fn_25313input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26268x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26614x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25375input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25437input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
&__inference_conv2d_layer_call_fn_26623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
A__inference_conv2d_layer_call_and_return_conditional_losses_26656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_1_layer_call_fn_26665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_2_layer_call_fn_26707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
3__inference_batch_normalization_layer_call_fn_26753
3__inference_batch_normalization_layer_call_fn_26766?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26784
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26802?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
8__inference_global_average_pooling2d_layer_call_fn_26807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_26813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
0__inference_time_distributed_layer_call_fn_26818
0__inference_time_distributed_layer_call_fn_26823?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
K__inference_time_distributed_layer_call_and_return_conditional_losses_26840
K__inference_time_distributed_layer_call_and_return_conditional_losses_26857?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
X
0
1
 2
!3
"4
#5
$6
%7"
trackable_list_wrapper
X
0
1
 2
!3
"4
#5
$6
%7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?states
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
#__inference_rnn_layer_call_fn_26878
#__inference_rnn_layer_call_fn_26899
#__inference_rnn_layer_call_fn_26920
#__inference_rnn_layer_call_fn_26941?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
>__inference_rnn_layer_call_and_return_conditional_losses_27115
>__inference_rnn_layer_call_and_return_conditional_losses_27289
>__inference_rnn_layer_call_and_return_conditional_losses_27463
>__inference_rnn_layer_call_and_return_conditional_losses_27637?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
forgetgate
?
inputgate1
?
inputgate2
?
outputgate"
_tf_keras_layer
 "
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_1_layer_call_fn_27650
5__inference_batch_normalization_1_layer_call_fn_27663?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27683
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27717?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_4_layer_call_fn_27726?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_4_layer_call_and_return_conditional_losses_27756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
__inference_call_25775x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_25830input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_conv2d_layer_call_fn_26623inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_conv2d_layer_call_and_return_conditional_losses_26656inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_1_layer_call_fn_26665inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26698inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_2_layer_call_fn_26707inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26740inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_batch_normalization_layer_call_fn_26753inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
3__inference_batch_normalization_layer_call_fn_26766inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26784inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26802inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_global_average_pooling2d_layer_call_fn_26807inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_26813inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
0__inference_time_distributed_layer_call_fn_26818inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
0__inference_time_distributed_layer_call_fn_26823inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
K__inference_time_distributed_layer_call_and_return_conditional_losses_26840inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
K__inference_time_distributed_layer_call_and_return_conditional_losses_26857inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_rnn_layer_call_fn_26878inputs/0"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_rnn_layer_call_fn_26899inputs/0"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_rnn_layer_call_fn_26920inputs"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_rnn_layer_call_fn_26941inputs"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_rnn_layer_call_and_return_conditional_losses_27115inputs/0"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_rnn_layer_call_and_return_conditional_losses_27289inputs/0"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_rnn_layer_call_and_return_conditional_losses_27463inputs"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_rnn_layer_call_and_return_conditional_losses_27637inputs"?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
X
0
1
 2
!3
"4
#5
$6
%7"
trackable_list_wrapper
X
0
1
 2
!3
"4
#5
$6
%7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_ourlstm_layer_call_fn_27783?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_ourlstm_layer_call_and_return_conditional_losses_27826?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_1_layer_call_fn_27650inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_1_layer_call_fn_27663inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27683inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27717inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_4_layer_call_fn_27726inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_4_layer_call_and_return_conditional_losses_27756inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_ourlstm_layer_call_fn_27783inputsstates/0states/1"?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_ourlstm_layer_call_and_return_conditional_losses_27826inputsstates/0states/1"?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
,:*02Adam/conv2d/kernel/m
:02Adam/conv2d/bias/m
.:,002Adam/conv2d_1/kernel/m
 :02Adam/conv2d_1/bias/m
.:,002Adam/conv2d_2/kernel/m
 :02Adam/conv2d_2/bias/m
,:*02 Adam/batch_normalization/gamma/m
+:)02Adam/batch_normalization/beta/m
/:-82Adam/rnn/ourlstm/dense/kernel/m
):'2Adam/rnn/ourlstm/dense/bias/m
1:/82!Adam/rnn/ourlstm/dense_1/kernel/m
+:)2Adam/rnn/ourlstm/dense_1/bias/m
1:/82!Adam/rnn/ourlstm/dense_2/kernel/m
+:)2Adam/rnn/ourlstm/dense_2/bias/m
1:/82!Adam/rnn/ourlstm/dense_3/kernel/m
+:)2Adam/rnn/ourlstm/dense_3/bias/m
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
,:*02Adam/conv2d/kernel/v
:02Adam/conv2d/bias/v
.:,002Adam/conv2d_1/kernel/v
 :02Adam/conv2d_1/bias/v
.:,002Adam/conv2d_2/kernel/v
 :02Adam/conv2d_2/bias/v
,:*02 Adam/batch_normalization/gamma/v
+:)02Adam/batch_normalization/beta/v
/:-82Adam/rnn/ourlstm/dense/kernel/v
):'2Adam/rnn/ourlstm/dense/bias/v
1:/82!Adam/rnn/ourlstm/dense_1/kernel/v
+:)2Adam/rnn/ourlstm/dense_1/bias/v
1:/82!Adam/rnn/ourlstm/dense_2/kernel/v
+:)2Adam/rnn/ourlstm/dense_2/bias/v
1:/82!Adam/rnn/ourlstm/dense_3/kernel/v
+:)2Adam/rnn/ourlstm/dense_3/bias/v
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v?
 __inference__wrapped_model_23809? !"#$%()'&*+E?B
;?8
6?3
input_1&??????????????????
? "@?=
;
output_1/?,
output_1???????????????????
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25375? !"#$%()'&*+U?R
;?8
6?3
input_1&??????????????????
?

trainingp "2?/
(?%
0??????????????????
? ?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_25437? !"#$%()'&*+U?R
;?8
6?3
input_1&??????????????????
?

trainingp"2?/
(?%
0??????????????????
? ?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26268? !"#$%()'&*+O?L
5?2
0?-
x&??????????????????
?

trainingp "2?/
(?%
0??????????????????
? ?
I__inference_basic_cnn_lstm_layer_call_and_return_conditional_losses_26614? !"#$%()'&*+O?L
5?2
0?-
x&??????????????????
?

trainingp"2?/
(?%
0??????????????????
? ?
.__inference_basic_cnn_lstm_layer_call_fn_24836? !"#$%()'&*+U?R
;?8
6?3
input_1&??????????????????
?

trainingp "%?"???????????????????
.__inference_basic_cnn_lstm_layer_call_fn_25313? !"#$%()'&*+U?R
;?8
6?3
input_1&??????????????????
?

trainingp"%?"???????????????????
.__inference_basic_cnn_lstm_layer_call_fn_25883? !"#$%()'&*+O?L
5?2
0?-
x&??????????????????
?

trainingp "%?"???????????????????
.__inference_basic_cnn_lstm_layer_call_fn_25936? !"#$%()'&*+O?L
5?2
0?-
x&??????????????????
?

trainingp"%?"???????????????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27683|()'&@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27717|()'&@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
5__inference_batch_normalization_1_layer_call_fn_27650o()'&@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
5__inference_batch_normalization_1_layer_call_fn_27663o()'&@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26784?Z?W
P?M
G?D
inputs8????????????????????????????????????0
p 
? "L?I
B??
08????????????????????????????????????0
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26802?Z?W
P?M
G?D
inputs8????????????????????????????????????0
p
? "L?I
B??
08????????????????????????????????????0
? ?
3__inference_batch_normalization_layer_call_fn_26753?Z?W
P?M
G?D
inputs8????????????????????????????????????0
p 
? "??<8????????????????????????????????????0?
3__inference_batch_normalization_layer_call_fn_26766?Z?W
P?M
G?D
inputs8????????????????????????????????????0
p
? "??<8????????????????????????????????????0?
__inference_call_25775? !"#$%()'&*+??<
5?2
0?-
x&??????????????????
? "%?"???????????????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26698?D?A
:?7
5?2
inputs&??????????????????0
? ":?7
0?-
0&??????????????????0
? ?
(__inference_conv2d_1_layer_call_fn_26665yD?A
:?7
5?2
inputs&??????????????????0
? "-?*&??????????????????0?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26740?D?A
:?7
5?2
inputs&??????????????????0
? ":?7
0?-
0&??????????????????0
? ?
(__inference_conv2d_2_layer_call_fn_26707yD?A
:?7
5?2
inputs&??????????????????0
? "-?*&??????????????????0?
A__inference_conv2d_layer_call_and_return_conditional_losses_26656?D?A
:?7
5?2
inputs&??????????????????
? ":?7
0?-
0&??????????????????0
? ?
&__inference_conv2d_layer_call_fn_26623yD?A
:?7
5?2
inputs&??????????????????
? "-?*&??????????????????0?
B__inference_dense_4_layer_call_and_return_conditional_losses_27756v*+<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
'__inference_dense_4_layer_call_fn_27726i*+<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_26813?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling2d_layer_call_fn_26807wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
B__inference_ourlstm_layer_call_and_return_conditional_losses_27826? !"#$%|?y
r?o
 ?
inputs?????????0
K?H
"?
states/0?????????
"?
states/1?????????
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
'__inference_ourlstm_layer_call_fn_27783? !"#$%|?y
r?o
 ?
inputs?????????0
K?H
"?
states/0?????????
"?
states/1?????????
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
>__inference_rnn_layer_call_and_return_conditional_losses_27115? !"#$%S?P
I?F
4?1
/?,
inputs/0??????????????????0

 
p 

 

 
? "2?/
(?%
0??????????????????
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_27289? !"#$%S?P
I?F
4?1
/?,
inputs/0??????????????????0

 
p

 

 
? "2?/
(?%
0??????????????????
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_27463? !"#$%L?I
B??
-?*
inputs??????????????????0

 
p 

 

 
? "2?/
(?%
0??????????????????
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_27637? !"#$%L?I
B??
-?*
inputs??????????????????0

 
p

 

 
? "2?/
(?%
0??????????????????
? ?
#__inference_rnn_layer_call_fn_26878? !"#$%S?P
I?F
4?1
/?,
inputs/0??????????????????0

 
p 

 

 
? "%?"???????????????????
#__inference_rnn_layer_call_fn_26899? !"#$%S?P
I?F
4?1
/?,
inputs/0??????????????????0

 
p

 

 
? "%?"???????????????????
#__inference_rnn_layer_call_fn_26920 !"#$%L?I
B??
-?*
inputs??????????????????0

 
p 

 

 
? "%?"???????????????????
#__inference_rnn_layer_call_fn_26941 !"#$%L?I
B??
-?*
inputs??????????????????0

 
p

 

 
? "%?"???????????????????
#__inference_signature_wrapper_25830? !"#$%()'&*+P?M
? 
F?C
A
input_16?3
input_1&??????????????????"@?=
;
output_1/?,
output_1???????????????????
K__inference_time_distributed_layer_call_and_return_conditional_losses_26840?L?I
B??
5?2
inputs&??????????????????0
p 

 
? "2?/
(?%
0??????????????????0
? ?
K__inference_time_distributed_layer_call_and_return_conditional_losses_26857?L?I
B??
5?2
inputs&??????????????????0
p

 
? "2?/
(?%
0??????????????????0
? ?
0__inference_time_distributed_layer_call_fn_26818uL?I
B??
5?2
inputs&??????????????????0
p 

 
? "%?"??????????????????0?
0__inference_time_distributed_layer_call_fn_26823uL?I
B??
5?2
inputs&??????????????????0
p

 
? "%?"??????????????????0