# Makefile for the universal dynamics experiment.

experiment=line

.SILENT:

# Front-facing targets
setup:
	make setup_${experiment}
	make clean

deploy:
	make deploy_${experiment}
	make clean

clean:
	rm *.rb


# Backend targets
setup_line:
	cp src/build_line_experiment.rb .
	ruby build_line_experiment.rb setup

setup_circle:
	cp src/build_circle_experiment.rb .
	ruby build_circle_experiment.rb setup

setup_abalone:
	cp src/build_abalone_experiment.rb .
	ruby build_abalone_experiment.rb setup

setup_mpg:
	cp src/build_mpg_experiment.rb .
	ruby build_mpg_experiment.rb setup

setup_mnist_dense:
	cp src/build_mnist_dense_experiment.rb .
	ruby build_mnist_dense_experiment.rb setup

setup_mnist_conv:
	cp src/build_mnist_conv_experiment.rb .
	ruby build_mnist_conv_experiment.rb setup

setup_cifar_dense:
	cp src/build_cifar_dense_experiment.rb .
	ruby build_cifar_dense_experiment.rb setup

setup_cifar_conv:
	cp src/build_cifar_conv_experiment.rb .
	ruby build_cifar_conv_experiment.rb setup

deploy_line:
	cp src/build_line_experiment.rb .
	ruby build_line_experiment.rb deploy

deploy_circle:
	cp src/build_circle_experiment.rb .
	ruby build_circle_experiment.rb deploy

deploy_abalone:
	cp src/build_abalone_experiment.rb .
	ruby build_abalone_experiment.rb deploy

deploy_mpg:
	cp src/build_mpg_experiment.rb .
	ruby build_mpg_experiment.rb deploy

deploy_mnist_dense:
	cp src/build_mnist_dense_experiment.rb .
	ruby build_mnist_dense_experiment.rb deploy

deploy_mnist_conv:
	cp src/build_mnist_conv_experiment.rb .
	ruby build_mnist_conv_experiment.rb deploy

deploy_cifar_dense:
	cp src/build_cifar_dense_experiment.rb .
	ruby build_cifar_dense_experiment.rb deploy

deploy_cifar_conv:
	cp src/build_cifar_conv_experiment.rb .
	ruby build_cifar_conv_experiment.rb deploy
